% This scripts tests a pre-trained networks the weights for which are saved in
% the SavedMat folder

clear
close all

load('Cochleograms/Input.mat');
load('SavedMat/W_RRN.mat');

%% NETWORK PARAMETERS
numIn    = size(Input(1,1,1).cochleogram, 1);
numEx    = 4000;     % Number of rate units
numOut   = 3;
tmax     = 2650;     % Maximum trial duration in ms
tstart   = 100;      % when to start the cochleogram input in ms

tau   = 25;                    % Unit time constant in ms
InAmp = 5;                     % Amplitude of input signal
noiseAmp = 0.0;

% Initialize variables for simulation
ExV = zeros(numEx,1);
Ex = zeros(numEx, 1);
Out = zeros(numOut, 1);
INPUT = zeros(numIn, tmax);
historyEx = zeros(numEx, tmax);
historyOut = zeros(numOut, tmax);

SAVESWITCH = 1; % Save network and output trajectories

%% NETWORK SIMLATION
sCnt = 0;
for subj = [1 2 5 6 7] % subject loop
    sCnt = sCnt + 1;
    sInd = find(CochleogramSubjects == subj);
    
    dCnt = 0;
    for digit = [0:9] %digit loop
        dCnt = dCnt + 1;
        dInd = find(CochleogramDigits == digit);
        
        for utter = 1:10 % utterance loop
            
            uInd = find(CochleogramUtterances == utter);
            
            % Reset relevant variables and histories
            ExV = ExV*0;
            Ex  = Ex*0;
            Out = Out*0;
            historyOut = historyOut*0;
            historyEx = historyEx*0;
            
            % Setup input from cochleoogram
            INPUT = INPUT*0;
            INPUT(:,tstart:tstart+Input(sInd,dInd,uInd).duration-1) = Input(sInd,dInd,uInd).cochleogram*InAmp;
            
            noise = randn(numEx, tmax)*noiseAmp;
            
            ExV = 2*rand(numEx,1)-1; % Set initial state randomly
            
            for t=1:tmax % time loop
                
                % Simulate trajectory and output
                ex_input = WExEx*Ex + WInEx*INPUT(:,t) + noise(:,t);
                
                ExV = ExV + (-ExV + ex_input)./tau;
                Ex = tanh(ExV);
                
                Out = WExOut*Ex;
                
                % Save trajectories
                historyEx(:,t)  = Ex;
                historyOut(:,t) = Out;
            end
            
            % Feedback to screen
            str = sprintf('Testing | (%d/%d/%d)',subj,digit,utter);
            if SAVESWITCH == 1
                % Save trajectories and outputs to disk during test phase
                saveHistories(sprintf('dataTest_%d_%d_%d', subj, digit, utter), historyEx,historyOut);
            end
            disp(str)
        end
    end
end

% Plot input, network trajectories and output during test phase            
sCnt = 0;
for subj = [1 2 5 6 7] % subject loop
    sCnt = sCnt + 1;
    sInd = find(CochleogramSubjects == subj);
    
    dCnt = 0;
    for digit = [0:9] %digit loop
        dCnt = dCnt + 1;
        dInd = find(CochleogramDigits == digit);
        
        for utter = 1:10 % utterance loop
            
            uInd = find(CochleogramUtterances == utter);
            load(sprintf('dataTest_%d_%d_%d', subj, digit, utter));

            figure(1);
            clf
            SP1 = subplot(2,2,1);
            SP2 = subplot(2,2,2);
            SP3 = subplot(2,2,3);
            SP4 = subplot(2,2,4);
            
            subplot(SP1);
            imagesc(historyEx);
            axis xy
            
            subplot(SP2);
            plot(historyOut','linewidth',2);
            
            subplot(SP3);
            imagesc(flipud(Input(sInd,dInd,uInd).cochleogram))
            axis xy
            
            subplot(SP4);
            downInds = (tstart-1)+find(historyOut(3,tstart:end) > 0.5);
            plot(historyOut(1,downInds), historyOut(2,downInds),'ro')
            xlim([-1,1])
            ylim([-1,1])
            drawnow;
            waitforbuttonpress
        end
    end
end