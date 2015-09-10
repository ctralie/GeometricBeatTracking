%Programmer: Chris Tralie
function [bts] = makeBeatsAudio( X, Fs, bts, outfilename )
    if ischar(bts)
        %Load ground truth from dataset
        fin = fopen(bts, 'r');
        bts = textscan(fin, '%f', 'delimiter', ' ');
        fclose(fin);    
        bts = bts{1};
        cutoff = bts(2:end) - bts(1:end-1);
        idx = find(cutoff < 0, 1);
        bts = bts(1:idx);
    end
    blip = cos(2*pi*440*(1:200)/Fs);
    for ii = 1:length(bts)
        idx = round(bts(ii)*Fs);
        X(idx:idx+length(blip)-1) = blip;
    end
    audiowrite(outfilename, X, Fs);
end

