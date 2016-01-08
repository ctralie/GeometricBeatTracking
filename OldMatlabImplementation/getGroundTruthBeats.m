%Return a cell array of all of the ground truth beats
function [allbts] = getGroundTruthBeats( filename )
    fin = fopen(filename, 'r');
    bts = textscan(fin, '%f', 'delimiter', ' ');
    fclose(fin);    
    bts = bts{1};
    cutoff = bts(2:end) - bts(1:end-1);
    idx = find(cutoff < 0);
    allbts = cell(1, length(idx));
    allbts{1} = bts(1:idx(1));
    for ii = 2:length(idx)
        allbts{ii} = bts(idx(ii-1)+1:idx(ii));
    end
end

