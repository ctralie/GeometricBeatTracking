function [ envfs, AOrig, Y ] = getFilteredOnsets( envorig, W, NPCs, NIters )
    env = envorig;
    envfs = cell(1, NIters);
    for kk = 1:NIters
        envd = get1DDelayEmbedding(env, W, 1);
        %TODO: Use pseudospectrum instead
        [A, Y, latent] = pca(envd);
        AOrig = A;
        AFFT = abs(fft(AOrig(:, 1:NPCs)));
        [~, idx] = max(AFFT, [], 1);
        [~, idx] = sort(idx);
        idx = idx(1:NPCs)

        A = zeros(size(AOrig));
        A(:, idx) = AOrig(:, idx);
        Y = bsxfun(@plus, (A*A'*Y')', mean(envd, 1));
        env = nan*ones(length(envorig), W);
        for ii = 1:W
            env(ii:size(Y, 1)+ii-1, ii) = Y(:, ii);
        end
        env = nanmean(env, 2);
        env = env - min(env);
        envfs{kk} = env/max(env);
    end
end

