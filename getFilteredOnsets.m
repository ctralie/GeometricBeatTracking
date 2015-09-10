function [ env, envf, SampleDelays, AOrig ] = getFilteredOnsets( X, Fs, NPCs )
    W = 250;
    NSines = 1;
    [~, env] = onsetenv(X, Fs);
    envd = get1DDelayEmbedding(env, W, 0);
    envdmag = sqrt(sum(envd.^2, 2));
    envd = bsxfun(@times, 1./envdmag, envd);
    [A, Y, latent] = pca(envd);
    AOrig = A;
    AFFT = abs(fft(AOrig(:, 1:NPCs)));
    [~, idx] = max(AFFT, [], 1);
    [~, idx] = sort(idx);
    idx(1:2*NSines)
    
    A = zeros(size(AOrig));
    A(:, idx(1:2*NSines)) = AOrig(:, idx(1:2*NSines));
    Y = bsxfun(@plus, (A*A'*Y')', mean(envd, 1));
    envf = nan*ones(length(env), W);
    for ii = 1:W
        envf(ii:size(Y, 1)+ii-1, ii) = Y(:, ii);
    end
    scale = sum(latent)/sum(latent(1:NPCs))*max(envdmag);
    envf = nanmean(envf, 2)*scale;
    SampleDelays = (1:length(envf))/W;
end

