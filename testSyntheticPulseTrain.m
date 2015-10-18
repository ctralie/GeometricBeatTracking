rng(110);
NPeriods = 120;
W = 250;
NPCs = 2;
SamplesPerPeriod = 120;
ChangeFac = 0.05*SamplesPerPeriod;
N = NPeriods*SamplesPerPeriod;
X = zeros(N, 1);
ii = 1;
T = SamplesPerPeriod;
idxs = [];
while ii < N
    ii = ii + T;
    if ii > N || ii < 1
        break
    end
    X(ii) = rand(1);
    idxs(end+1) = ii;
    T = round(T + exprnd(ChangeFac) - ChangeFac);
end
X = X + 0.1*randn(size(X));
X = X - min(X);

[envfs, AOrig, Y] = getFilteredOnsets(X, W, NPCs, 4);
clf;
plot(X, 'b');
hold on;
plot(envfs{end}, 'r');