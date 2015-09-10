init;
filename = 'train5';
soundfilename = sprintf('examples1/%s.wav', filename);
[X, Fs] = audioread(soundfilename);
allbts = getGroundTruthBeats(sprintf('examples1/%s.txt', filename));

[env, envf, SampleDelays, AOrig, Y] = getFilteredOnsets(X, Fs, 10);
figure(1);
clf;
subplot(2, 1, 1);
plot(SampleDelays, env);
title(sprintf('Original Envelope %s', filename));
xlabel('Seconds');
subplot(2, 1, 2);
plot(SampleDelays, envf);
hold on;
bts = allbts{1};
stem(bts, ones(size(bts))*mean(abs(envf)));
xlabel('Seconds');
title('Low Dimensional Projection with Ground Truth Beats');
figure(2);
k = 20;
imagesc(AOrig(:, 1:k));
title(sprintf('First %i Principal Components %s', k, filename));

%Export for viewing in 2d function viewer
X = [1:length(env); env(:)']';
save(sprintf('%senv.mat', filename), 'Fs', 'SampleDelays', 'X', 'soundfilename');
X = [1:length(envf); envf(:)']';
save(sprintf('%senvf.mat', filename), 'Fs', 'SampleDelays', 'X', 'soundfilename');