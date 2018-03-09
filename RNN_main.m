% This is the main script to train and test networks from scratch
% It sets up the network parameters, simulation parameters, training and
% testing parameters and calls the simulateRNN script

clear
close all

%% SIMULATION PARAMETERS
numLabs  = 3;        % Number of processors to use
SEED = 777;          % Simulation seed
TEMPSEED = 9999;     % to help use different seed when collecting innate traj vs training vs testing


%% TRAIN/TEST STIMULI
trainSubjects = [1]; %[1, 5, 6]; % List of subjects to train on
trainDigits = [0:9]; % List of digits to train on
trainUtterances{1} = [1 7 10; 1 9 10; 1 5 10; 1 7 10; 1 3 7; 1 8 10; 1 5 6; 2 1 9; 3 1 10; 1 5 10];  % List of Utterances to train on
%trainUtterances{2} = [1 10 7; 9 3 6; 8 10 4; 1 10 5; 3 10 5; 9 8 5; 3 6 1; 10 3 2; 3 2 5; 5 10 3]; % Training list per subject - this list is for subject 5
%trainUtterances{3} = [4 10 3; 1 2 6; 5 2 8; 5 2 7; 3 2 9; 3 2 6; 5 2 6; 4 10 7; 1 10 7; 5 8 2];
testSubjects = [1]; %[1, 2, 5, 6, 7]; % List of subjects to test on
testDigits = [0:9]; % List of digits to test on
testUtterances{1} = [1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10]; % List of utterances to test on
%testUtterances{2} = [1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10];
%testUtterances{3} = [1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10];
%testUtterances{4} = [1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10];
%testUtterances{5} = [1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10; 1:10];

% Load cochleograms and output (transcription) targets
load('Cochleograms/Input.mat');
load('Transcriptions/OutTarget.mat');


%% NETWORK PARAMETERS
numIn    = size(Input(1,1,1).cochleogram, 1);
numEx    = 2100; %4000;   % Number of rate units
numOut   = 3;
tmax     = 2650;     % Maximum trial duration in ms
tstart   = 100;      % when to start the cochleogram input in ms

connectionProbability = 0.2;   % Connection probability between rate units
tau   = 25;                    % Unit time constant in ms

g = 1.6;                       % initial value for gain of the network
InAmp = 5;                     % Amplitude of input signal
alpha = 10;                    % RLS parameter
fractionUnitsTrained = 0.9;    % Fraction of rate units trained to match the target trajectory


%% SIMULATION PHASES AND PARAMETERS
Runphase = struct('collectInnate',1,'trainRecurrent',2,'trainOutput',3,'test',4);
phases = [Runphase.collectInnate Runphase.trainRecurrent Runphase.trainOutput Runphase.test];
% Number of trials to loop through each condition
phaseParam(Runphase.collectInnate).numTrialsPerCondition = 1;
phaseParam(Runphase.trainRecurrent).numTrialsPerCondition = 130;
phaseParam(Runphase.trainOutput).numTrialsPerCondition = 25;
phaseParam(Runphase.test).numTrialsPerCondition = 1;
% Amplitude of noise to inject into recurrent units
phaseParam(Runphase.collectInnate).noise = 0.0;
phaseParam(Runphase.trainRecurrent).noise = 0.05;
phaseParam(Runphase.trainOutput).noise = 0.05;
phaseParam(Runphase.test).noise = 0.0;
% Inputs to train/test with
phaseParam(Runphase.collectInnate).subjects = trainSubjects;
phaseParam(Runphase.collectInnate).digits = trainDigits;
phaseParam(Runphase.collectInnate).utterances = trainUtterances;
phaseParam(Runphase.trainRecurrent).subjects = trainSubjects;
phaseParam(Runphase.trainRecurrent).digits = trainDigits;
phaseParam(Runphase.trainRecurrent).utterances = trainUtterances;
phaseParam(Runphase.trainOutput).subjects = trainSubjects;
phaseParam(Runphase.trainOutput).digits = trainDigits;
phaseParam(Runphase.trainOutput).utterances = trainUtterances;
phaseParam(Runphase.test).subjects = testSubjects;
phaseParam(Runphase.test).digits = testDigits;
phaseParam(Runphase.test).utterances = testUtterances;

%% RUN SIMULATIONS IN PARALLEL
poolobj = parpool(numLabs);
for phase = phases
    simulateRNN
end
delete(poolobj);
