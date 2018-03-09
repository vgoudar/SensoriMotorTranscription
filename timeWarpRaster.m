function warpedRaster  =  timeWarpRaster(RASTER,TimeScaleFactor,startWindow, endWindow, sExt, motorRaster)
% This function performs the actual time warping of innate trajectories
% It is called from within the warpTrajectories function

HighResSampleRate = 100;           %upsample trajectory to 100 times the sampling rate (e.g., 0.01 ms)

Window = [startWindow endWindow];
Data = RASTER(:,Window(1):Window(2));

NumPoint = diff(Window);
highresT = 0:1/HighResSampleRate:NumPoint;

iData = interp1(0:NumPoint,Data(:,:)',highresT);  % Upsample sensory trajectory only
iData = iData';

newTSample = round([0:TimeScaleFactor:NumPoint]*HighResSampleRate); % Scale sensory trajectory to appropriate duration
nData = iData(:,newTSample+1);
warpedRaster = [RASTER(:,1:Window(1)-1) nData RASTER(:,(Window(2)+1):(Window(2)+sExt)) motorRaster]; % Concatenate common motor trajectory

end

