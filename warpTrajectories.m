function EXTARGET = warpTrajectories(phaseParam, EXTARGET, tstart)
% This function warps all sensory trajectories to a subject-spectific templates
% A common motor trajectory is chosen per digit (across subjects) from the
% template trajectory of the first subject
% It is called from the simulateRNN script
% It uses the helper function timeWarpRaster

    load('Cochleograms/Input.mat');

    dCnt = 0;
    for digit = phaseParam.digits
        dCnt = dCnt + 1;
        dInd = find(CochleogramDigits == digit);

        sCnt = 0;
        for subj = phaseParam.subjects
            sCnt = sCnt + 1;
            sInd = find(CochleogramSubjects == subj);

            ds = [];
            for utter = phaseParam.utterances{sCnt}(dCnt,:)
                uInd = find(CochleogramUtterances == utter);
                ds = [ds Input(sInd,dInd,uInd).duration];
            end
            [~, mid] = sort(ds);
            % Find subject-specific template trajectory as the one with median duration
            midU = phaseParam.utterances{sCnt}(dCnt,mid(ceil(length(phaseParam.utterances{sCnt}(dCnt,:))/2)));
            midU = find(CochleogramUtterances == phaseParam.utterances{sCnt}(dCnt,midU));

            Raster = EXTARGET(sInd,dInd,midU).extarget;
            if subj == phaseParam.subjects(1)
                % Save single motor template for all subjects and utterances
                motorRast = Raster(:,(tstart+Input(sInd,dInd,midU).duration+151):end);
                Raster2 = Raster;
            else
                Raster2 = [Raster(:,1:(tstart+Input(sInd,dInd,midU).duration+150)) motorRast];
            end
            EXTARGET(sInd,dInd,midU).extarget = Raster2;
            figure(2);
            clf            
            plot(Raster2(1,:))
            hold on;

            % Warp template sensory trajectories for each utterance and concatenate common motor template 
            for utter = phaseParam.utterances{sCnt}(dCnt,:)
                uInd = find(CochleogramUtterances == utter);
                if uInd == midU
                    continue;
                end
                TimeScaleFactor = Input(sInd,dInd,midU).duration/Input(sInd,dInd,uInd).duration;
                fprintf('Warping EXTARGETS: S=%d; D=%d; U=%d [Stand=%4d Org=%d TimeScaleFactor=%4.2f]\n',subj,digit,utter,Input(sInd,dInd,midU).duration,Input(sInd,dInd,uInd).duration,TimeScaleFactor);
                EXTARGET(sInd,dInd,uInd).extarget = timeWarpRaster(Raster,TimeScaleFactor,tstart,tstart+Input(sInd,dInd,midU).duration, 150, motorRast);
                plot(EXTARGET(sInd,dInd,uInd).extarget(1,:))
            end
            drawnow
        end
    end
end
