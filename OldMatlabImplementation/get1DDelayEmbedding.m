function [ Y ] = get1DDelayEmbedding( X, M, Normalize )
    if nargin < 3
        Normalize = 0;
    end
    N = length(X);
    Y = zeros(N-M+1, M);
    for ii = 1:size(Y, 1)
        y = X(ii:ii+M-1);
        Y(ii, :) = y(:)';
    end
    if Normalize
        Y = bsxfun(@times, 1./sqrt(sum(Y.^2, 2)), Y);
    end
end

