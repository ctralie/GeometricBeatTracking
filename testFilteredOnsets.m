init;
[X, Fs] = audioread('examples1/train4.wav');
[env, envf, SampleDelays, AOrig] = getFilteredOnsets(X, Fs, 10);
subplot(1, 2, 1);
plot(SampleDelays, envf);
subplot(1, 2, 2);
J = morseFiltration(envf);
plotpersistencediagram(J);


% [pks, locs] = findpeaks(-envf);
% bts = SampleDelays(locs);
% bts = bts(1:2:end);
% tempo = 60/mean(bts(2:end)-bts(1:end-1))
% makeBeatsAudio( X, Fs, bts, 'train1mybeats.wav' );
% 
% clf;
% subplot(2, 1, 1);
% plot(SampleDelays, env, 'b');
% hold on;
% plot(SampleDelays, envf*5, 'r');
% subplot(2, 1, 2);
% plot(SampleDelays, envf, 'r');
% hold on;
% stem(SampleDelays(locs), envf(locs));