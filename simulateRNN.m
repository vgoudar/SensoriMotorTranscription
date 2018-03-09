% Script to collect innate trajectories, train and test the network
% This script is called from the RNN_main script
% It uses the helper functions warpTrajectories and saveHistories
tic

clear historyEx historyOut
clear WExEx WInEx WExOut
clear P PRec PreSyn
clear EXTARGET

CellsPerLab = numEx/numLabs; % Number of cells per worker - number of units should be exactly divisible by number of workers!!!
fprintf('numLabs=%4d; numEx=%4d, CellsPerLab=%4d\n',numLabs,numEx,CellsPerLab);

%%               WEIGHTS INITIALIZATION
if phase == Runphase.collectInnate
    rand('seed',SEED);
    randn('seed',SEED);
    
    % Input Weights:
    sizeColumn =  floor(numEx/numIn);
    WInEx = randn(numIn,numEx);
    WInExMask = zeros(numIn,numEx);
    for i = 1:numIn
        WInExMask(i,1+(i-1)*sizeColumn:sizeColumn+(i-1)*sizeColumn) = 1;
    end
    WInEx = WInEx.*WInExMask;
    WInEx = WInEx';
    
    % Recurrent Weights:
    WMask = rand(numEx,numEx);        
    WMask(WMask<(1.0-connectionProbability))=0; % sparsify recurrent weight matrix
    WMask(WMask>0) = 1;
    WExEx = randn(numEx,numEx)*sqrt(1/(numEx*connectionProbability));
    WExEx = WExEx.*WMask*g;
    WExEx(1:(numEx+1):numEx*numEx)=0; % zero out autapses
    
    % Output Weights:
    WExOut = randn(numOut,numEx)*sqrt(1/numEx);
else
    load W_RRN;
end

% Distribute matrices across workers
wexex  = Composite();
winex  = Composite();
wexout = Composite();

for i=1:numLabs
    winex{i} = WInEx((1+(i-1)*CellsPerLab):(CellsPerLab*(i)),:);
    wexex{i} = WExEx((1+(i-1)*CellsPerLab):(CellsPerLab*(i)),:);
end
wexout{1} = WExOut(:,:);

WExEx = wexex;
WInEx = winex;
WExOut = wexout;

if phase == Runphase.trainRecurrent
    % Load target trajectories for recurrent net training and distribute across workers
    load('ExTarget.mat');
    extarget = Composite();
    for i =1:numLabs
        extarget{i} = EXTARGET;
    end
    EXTARGET = extarget;
elseif phase == Runphase.trainOutput
    % Initialize P matrix (RLS) for output training
    P = Composite();
    P{1} = eye(numEx)/alpha;
end
%%%%%%%%%%%%%%%%%%%%% END WEIGHTS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%

tempseed = SEED+TEMPSEED*phase;
noiseAmp = phaseParam(phase).noise;

%%                     PARALLEL SIMULATION
spmd
    rand('seed',tempseed+labindex);
    randn('seed',tempseed+labindex);
    
    % Initialize variables for simulation - rate units are distributed across workers
    INPUT = zeros(numIn,tmax);
    ExV      = zeros(CellsPerLab,1);
    ex_input = zeros(CellsPerLab,1);
    ExLocal  = zeros(CellsPerLab,1);
    Ex       = zeros(numEx,1);
    historyExLocal = zeros(CellsPerLab,tmax);
    historyEx      = zeros(numEx,tmax);
    if phase == Runphase.collectInnate
        % Subject-specific common initial state for utterances of each digit
        ExVInitState = zeros(CellsPerLab, length(phaseParam(phase).digits));
        ExVInitStateInitialized = boolean(zeros(length(phaseParam(phase).digits),1));
        EXTARGET = struct([]);
    elseif phase == Runphase.trainRecurrent
        trainCellList = sort(randsample(CellsPerLab, floor(CellsPerLab*fractionUnitsTrained)))'; % choose units that will be trained to match template
        historyRecErrLocal = zeros(CellsPerLab,tmax);
        historyRecErr = zeros(numEx,tmax);
        
        % Initialize P matrix (RLS) for recurrent network training
        PreSyn = struct([]);
        PRec = struct([]);
        for i=1:CellsPerLab
            PreSyn(i).ind= find(WExEx(i,:));
            PRec(i).P = eye(length(PreSyn(i).ind))/alpha;
        end
        
    elseif phase >= Runphase.trainOutput && labindex == 1
        % Initialize output variables for simulation
        Out = zeros(numOut,1);
        historyOut = zeros(numOut,tmax);
        if phase == Runphase.trainOutput
            OUTTARGET = zeros(numOut, tmax);
            historyOutErr = zeros(numOut, tmax);
        end
    end
    
    for loop=1:phaseParam(phase).numTrialsPerCondition % trial loop
        if phase == Runphase.trainRecurrent && loop >= 35
            noiseAmp = 0.5; % Start with low noise and increase it during recurrent training
        end
        
        sCnt = 0;
        for subj = phaseParam(phase).subjects % subject loop
            sCnt = sCnt + 1;
            sInd = find(CochleogramSubjects == subj);
            
            dCnt = 0;
            for digit = phaseParam(phase).digits % digit loop
                dCnt = dCnt + 1;
                dInd = find(CochleogramDigits == digit);
                
                for utter = phaseParam(phase).utterances{sCnt}(dCnt,:) % utterance loop
                    
                    uInd = find(CochleogramUtterances == utter);
                    
                    % Reset relevant variables and histories
                    historyExLocal = historyExLocal*0;
                    historyEx = historyEx*0;
                    ExV = ExV*0;
                    Ex  = Ex*0;
                    if phase == Runphase.trainRecurrent
                        historyRecErrLocal = historyRecErrLocal*0;
                        historyRecErr = historyRecErr*0;
                        
                    elseif phase >= Runphase.trainOutput && labindex == 1
                        Out = Out*0;
                        historyOut = historyOut*0;
                        if phase == Runphase.trainOutput
                            historyOutErr = historyOutErr*0;
                        end
                    end
                    
                    % Setup input from cochleoogram
                    INPUT = INPUT*0;
                    INPUT(:,tstart:tstart+Input(sInd,dInd,uInd).duration-1) = Input(sInd,dInd,uInd).cochleogram*InAmp;
                    
                    noise = randn(CellsPerLab, tmax)*noiseAmp;
                    
                    % Set up target (both recurrent and output)
                    if phase == Runphase.trainRecurrent
                        
                        odInd = find(OutDigits == digit);
                        oDur = size(OutTarget(odInd).transcription,1);
                        TRAIN_WINDOW = [tstart tstart+Input(sInd,dInd,uInd).duration + 300 + oDur + 100];
                        currentEXTARGET = EXTARGET(sInd,dInd,uInd).extarget;
                        EXTARGETLOCAL = currentEXTARGET((1+(labindex-1)*CellsPerLab):(CellsPerLab*(labindex)),:);
                        
                    elseif phase == Runphase.trainOutput && labindex == 1
                        odInd = find(OutDigits == digit);
                        TRAIN_WINDOW = [1 tmax];
                        OUTTARGET = OUTTARGET*0;
                        motorStart = tstart + Input(sInd,dInd,uInd).duration + 301;
                        motorEnd = motorStart + size(OutTarget(odInd).transcription,1) - 1;
                        OUTTARGET(1:2, motorStart:motorEnd) = OutTarget(odInd).transcription';
                        OUTTARGET(3, motorStart:motorEnd) = 1.0;
                        OUTTARGET(1:2, (motorStart-300):(motorStart-1)) = repmat(OutTarget(odInd).transcription(1,:)', 1, 300);
                        OUTTARGET(1:2, (motorEnd+1):tmax) = repmat(OutTarget(odInd).transcription(end,:)', 1, tmax-motorEnd);
                    end
                    
                    % Randomize initial state (except during collection of innate trajectories
                    if phase == Runphase.collectInnate
                        if ExVInitStateInitialized(dCnt)
                            ExV = ExVInitState(:,dCnt); % Common initial state per digit during innate trajectory collection
                        else
                            ExV = 2*rand(CellsPerLab,1)-1;
                            ExVInitState(:,dCnt) = ExV;
                            ExVInitStateInitialized(dCnt) = true;
                        end
                    else
                        ExV = 2*rand(CellsPerLab,1)-1;
                    end
                    
                    for t=1:tmax % time loop
                        
                        % Simulate trajectory
                        ex_input = WExEx*Ex + WInEx*INPUT(:,t) + noise(:,t);
                        
                        ExV = ExV + (-ExV + ex_input)./tau;
                        ExLocal = tanh(ExV);
                        
                        % Synchronize unit rates across workers
                        labBarrier;
                        Ex = gcat(ExLocal,1); 
                        labBarrier;
                        
                        % Compute output
                        if phase >= Runphase.trainOutput && labindex == 1
                            Out = WExOut*Ex;
                        end
                        
                        % Output training
                        if (phase == Runphase.trainOutput && labindex==1)
                            if t >= TRAIN_WINDOW(1) && t <= TRAIN_WINDOW(2)
                                error_out = Out - OUTTARGET(:,t);
                                kOut = P*Ex;
                                ExPEx = Ex'*kOut;
                                c = 1.0/(1.0 + ExPEx);
                                P = P - kOut*(kOut'*c);
                                dw = repmat(error_out,1,numEx).*repmat(kOut',numOut,1)*c;
                                WExOut = WExOut - dw;
                                
                                historyOutErr(:, t) = error_out;
                            end
                        end
                        
                        % Recurrent network training
                        if (phase == Runphase.trainRecurrent)
                            if t >= TRAIN_WINDOW(1) && t <= TRAIN_WINDOW(2) && mod(t,5) == 0
                                error_rec = ExLocal - EXTARGETLOCAL(:,t);
                                for i=trainCellList % Train synapses onto recurrent units chosen for training                                    
                                    preind = PreSyn(i).ind;
                                    ex =  Ex(preind);
                                    kRec=PRec(i).P*ex;
                                    expex = ex'*kRec;
                                    c = 1.0/(1.0 + expex);
                                    PRec(i).P = PRec(i).P - kRec*(kRec'*c);
                                    dw = error_rec(i)*kRec*c;
                                    WExEx(i, preind) = WExEx(i, preind) - dw';
                                end                                
                                historyRecErrLocal(:, t) = error_rec;                                
                            end
                        end
                        
                        % Save trajectories
                        historyExLocal(:,t)  = ExLocal;
                        if phase >= Runphase.trainOutput && labindex == 1
                            historyOut(:,t)      = Out;
                        end
                    end
                    
                    % Synchronize trajectories across workers
                    labBarrier;
                    historyEx = gcat(historyExLocal,1);
                    if (phase == Runphase.trainRecurrent)
                        historyRecErr = gcat(historyRecErrLocal, 1);
                    end
                    labBarrier;
                    
                    % Save innate trajectories
                    if (phase == Runphase.collectInnate && labindex == 1 && loop == 1)
                        EXTARGET(sInd,dInd,uInd).extarget  = historyEx;
                        EXTARGET(sInd,dInd,uInd).subject  = subj;
                        EXTARGET(sInd,dInd,uInd).digit  = digit;
                        EXTARGET(sInd,dInd,uInd).utterance  = utter;
                    end
                    
                    % Feedback to screen
                    if labindex == 1
                        if (phase == Runphase.collectInnate)
                            str = sprintf('Collecting Innate Trajectories | LOOP = %3d(%d/%d/%d)',loop,subj,digit,utter);
                        elseif (phase == Runphase.trainRecurrent)
                            tInd = TRAIN_WINDOW(1):TRAIN_WINDOW(2);
                            tInd = (TRAIN_WINDOW(1)-1)+find(mod(tInd,5)==0);
                            meanErr = mean(sqrt(diag(historyRecErr(:,tInd)'*historyRecErr(:,tInd))));
                            str = sprintf('Training Recurrent Network | LOOP = %d/%d(%d/%d/%d) | meanRecErr = %f',loop,phaseParam(phase).numTrialsPerCondition,subj,digit,utter,meanErr);
                        elseif  (phase == Runphase.trainOutput)
                            tInd = TRAIN_WINDOW(1):TRAIN_WINDOW(2);
                            meanErr = mean(sqrt(diag(historyOutErr(:,tInd)'*historyOutErr(:,tInd))));
                            str = sprintf('Trainint Output | LOOP = %d/%d(%d/%d/%d) | meanOutErr = %f',loop,phaseParam(phase).numTrialsPerCondition,subj,digit,utter,meanErr);
                        else
                            str = sprintf('Testing | LOOP = %3d(%d/%d/%d)',loop,subj,digit,utter);
                            % Save trajectories and outputs to disk during test phase
                            saveHistories(sprintf('data_%d_%d_%d_%d',loop, subj, digit, utter), historyEx,historyOut);
                        end
                        disp(str)
                    end
                    
                    
                end
            end
        end
    end
    
    % Synchronize weight matrices
    labBarrier;
    WExEx    = gcat(WExEx,1);
    WInEx    = gcat(WInEx,1);
    labBarrier;
    
end
%%%%%%%%%%%%%%%%%%%%% END PARALLEL SIMULATION %%%%%%%%%%%%%%%%%%%%%%%%%

% Store weight matrices at client
WExEx       = WExEx{1};
WInEx       = WInEx{1};
WExOut      = WExOut{1};

fprintf('Runtime = %12.6f\n',toc);

if phase == Runphase.collectInnate
    % Generate target trajectories and save them to disk
    EXTARGET    = EXTARGET{1};
    EXTARGET = warpTrajectories(phaseParam(phase), EXTARGET, tstart);
    save ExTarget EXTARGET
    
elseif phase == Runphase.test 
    % Plot input, network trajectories and output during test phase
    figure(1);
    clf
    SP1 = subplot(2,2,1);
    SP2 = subplot(2,2,2);
    SP3 = subplot(2,2,3);
    SP4 = subplot(2,2,4);
    for loop=1:phaseParam(phase).numTrialsPerCondition
        sCnt = 0;
        for subj = phaseParam(phase).subjects
            sCnt = sCnt + 1;
            sInd = find(CochleogramSubjects == subj);
            dCnt = 0;
            for digit = phaseParam(phase).digits
                dCnt = dCnt + 1;
                dInd = find(CochleogramDigits == digit);
                for utter = phaseParam(phase).utterances{sCnt}(dCnt,:)
                    uInd = find(CochleogramUtterances == utter);
                    load(sprintf('data_%d_%d_%d_%d',loop, subj, digit, utter));
                    drawnow;
                    
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
    end
end

% Save weight matrices to disk
save('W_RRN', 'WExOut', 'WExEx', 'WInEx', 'SEED');
