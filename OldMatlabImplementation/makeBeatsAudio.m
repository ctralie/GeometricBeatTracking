%Programmer: Chris Tralie
function [bts] = makeBeatsAudio( X, Fs, bts, outfilename )
    blip = cos(2*pi*440*(1:200)/Fs);
    for ii = 1:length(bts)
        idx = round(bts(ii)*Fs);
        X(idx:idx+length(blip)-1) = blip;
    end
    audiowrite(outfilename, X, Fs);
end

