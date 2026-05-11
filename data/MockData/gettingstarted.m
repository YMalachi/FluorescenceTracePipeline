%% load AOD dataset
load('data.1.train.preprocessed.mat')

%% plot two example cell

figure('Position',[100 100 800 300])
T = 10000;

subplot(2,1,1)
cell_idx = 1;
dt = 1/data{cell_idx}.fps;
time = dt:dt:dt*T;
plot(time,zscore(data{cell_idx}.calcium(1:T))), hold on
plot(time,double(data{cell_idx}.spikes(1:T))-4)
set(gca,'box','off')
title('Example with average SNR')

subplot(2,1,2)
cell_idx = 5;
dt = 1/data{cell_idx}.fps;
time = dt:dt:dt*T;
plot(time,zscore(data{cell_idx}.calcium(1:T))), hold on
plot(time,double(data{cell_idx}.spikes(1:T))-4)
xlabel('Time (s)')
set(gca,'box','off')
title('Example with high SNR')

