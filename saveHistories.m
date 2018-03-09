function saveHistories(fName, historyEx, historyOut)
% This functions saves trajectories to disk
% It is called from the simulateRNN script
    save(fName, 'historyEx', 'historyOut');
end