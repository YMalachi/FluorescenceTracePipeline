function result = run_mlspike_model(calcium, fps, params)
% run_mlspike_model
%
% Goal:
%   Run MLspike spike inference on one calcium fluorescence trace and
%   convert the estimated spike times into a binary predicted-events vector.
%
% Inputs:
%   calcium - calcium fluorescence trace
%   fps     - sampling rate in Hz
%   params  - struct with MLspike parameters:
%
%       params.use_autocalibration
%       params.a
%       params.tau
%       params.sigma
%       params.saturation
%       params.drift_parameter
%
%       params.amin
%       params.amax
%       params.taumin
%       params.taumax
%
% Output:
%   result - struct containing:
%       predicted_events
%       spike_times_sec
%       fit
%       drift
%       calcium_mlspike
%       params
%       parest
%       n_predicted_events
%
% Important:
%   MLspike expects calcium traces with a nonzero baseline.
%   Do not pass z-scored calcium directly.
%   This function prepares a positive, mean-normalized version of calcium.

    %% Basic input checks

    if nargin ~= 3
        error('run_mlspike_model requires exactly 3 inputs: calcium, fps, params.');
    end

    if isempty(calcium)
        error('Input calcium trace is empty.');
    end

    if isempty(fps) || ~isnumeric(fps) || ~isscalar(fps)
        error('fps must be a single numeric value.');
    end

    if fps <= 0
        error('fps must be positive. Received: %.4f', fps);
    end

    if ~isstruct(params)
        error('params must be a struct.');
    end

    required_fields = { ...
        'use_autocalibration', ...
        'a', ...
        'tau', ...
        'sigma', ...
        'saturation', ...
        'drift_parameter', ...
        'amin', ...
        'amax', ...
        'taumin', ...
        'taumax'};

    for i = 1:length(required_fields)
        field_name = required_fields{i};

        if ~isfield(params, field_name)
            error('params is missing required field: %s', field_name);
        end
    end

    %% Prepare calcium

    calcium = double(calcium(:));

    if any(isnan(calcium))
        error('Calcium trace contains NaN values.');
    end

    if any(isinf(calcium))
        error('Calcium trace contains Inf values.');
    end

    if std(calcium) == 0
        error('Calcium trace has zero standard deviation.');
    end

    % MLspike expects calcium with a nonzero baseline.
    % Our mock traces may be centered around zero or contain negative values,
    % so we shift them to be positive and normalize the mean to approximately 1.
    calcium_mlspike = prepare_calcium_for_mlspike(calcium);

    dt = 1 / fps;

    %% Create MLspike parameter structure

    % Prefer the parameter constructor used in the MLspike demo.
    % Fall back to spk_est('par') if tps_mlspikes is unavailable.
    if exist('tps_mlspikes', 'file')
        par = tps_mlspikes('par');
    else
        par = spk_est('par');
    end

    par.dt = dt;
    par.saturation = params.saturation;
    par.drift.parameter = params.drift_parameter;
    par.dographsummary = false;

    %% Optional autocalibration

    autocalibration_output = struct();
    autocalibration_output.used = logical(params.use_autocalibration);
    autocalibration_output.tau = NaN;
    autocalibration_output.a = NaN;
    autocalibration_output.sigma = NaN;

    if params.use_autocalibration

        try
            pax = spk_autocalibration('par');
            pax.dt = dt;

            pax.amin = params.amin;
            pax.amax = params.amax;
            pax.taumin = params.taumin;
            pax.taumax = params.taumax;
            pax.saturation = params.saturation;

            % Avoid summary plots during pipeline runs.
            pax.mlspikepar.dographsummary = false;

            [tau_est, a_est, sigma_est] = spk_autocalibration(calcium_mlspike, pax);

            par.a = a_est;
            par.tau = tau_est;
            par.finetune.sigma = sigma_est;

            autocalibration_output.tau = tau_est;
            autocalibration_output.a = a_est;
            autocalibration_output.sigma = sigma_est;

        catch ME
            error('MLspike autocalibration failed: %s', ME.message);
        end

    else

        par.a = params.a;
        par.tau = params.tau;
        par.finetune.sigma = params.sigma;
    end

    %% Run MLspike

    try
        [spike_times_raw, fit, drift, parest] = spk_est(calcium_mlspike, par);
    catch ME
        error('MLspike spk_est failed: %s', ME.message);
    end

    %% Convert MLspike output to spike times in seconds

    spike_times_sec = normalize_mlspike_spike_output(spike_times_raw, length(calcium), fps);

    %% Convert spike times to predicted event vector

    predicted_events = false(size(calcium));

    if ~isempty(spike_times_sec)

        spike_idx = round(spike_times_sec * fps) + 1;

        % Keep only valid indices.
        spike_idx = spike_idx(spike_idx >= 1 & spike_idx <= length(calcium));

        % Multiple inferred spikes in the same bin become one event.
        predicted_events(unique(spike_idx)) = true;
    end

    %% Package output

    result = struct();

    result.predicted_events = predicted_events;
    result.spike_times_sec = spike_times_sec;

    result.fit = fit;
    result.drift = drift;
    result.parest = parest;

    result.calcium_mlspike = calcium_mlspike;
    result.params = params;
    result.par_used = par;
    result.autocalibration = autocalibration_output;

    result.n_predicted_events = sum(predicted_events);
end

%% Local helper functions

function calcium_mlspike = prepare_calcium_for_mlspike(calcium)
% prepare_calcium_for_mlspike
%
% MLspike should not receive z-scored / mean-subtracted calcium.
% This helper shifts the trace to be positive and normalizes mean to 1.

    calcium = double(calcium(:));

    min_val = min(calcium);

    if min_val <= 0
        calcium_shifted = calcium - min_val + eps;
    else
        calcium_shifted = calcium;
    end

    mean_val = mean(calcium_shifted);

    if mean_val <= 0 || isnan(mean_val) || isinf(mean_val)
        error('Could not normalize calcium trace for MLspike.');
    end

    calcium_mlspike = calcium_shifted / mean_val;
end

function spike_times_sec = normalize_mlspike_spike_output(spike_times_raw, n_samples, fps)
% normalize_mlspike_spike_output
%
% MLspike usually returns estimated spike times in seconds.
% This helper also handles the case where the output is accidentally a
% full-length activity vector.

    if iscell(spike_times_raw)
        if isempty(spike_times_raw)
            spike_times_raw = [];
        else
            spike_times_raw = spike_times_raw{1};
        end
    end

    spike_times_raw = double(spike_times_raw(:));

    if isempty(spike_times_raw)
        spike_times_sec = [];
        return;
    end

    % If output length equals the trace length, interpret it as an activity
    % vector and convert nonzero entries to event times.
    if length(spike_times_raw) == n_samples
        event_idx = find(spike_times_raw > 0);
        spike_times_sec = (event_idx - 1) / fps;
    else
        % Standard MLspike case: vector of spike times in seconds.
        spike_times_sec = spike_times_raw;
    end

    % Remove invalid values.
    spike_times_sec = spike_times_sec(~isnan(spike_times_sec));
    spike_times_sec = spike_times_sec(~isinf(spike_times_sec));
    spike_times_sec = spike_times_sec(spike_times_sec >= 0);
end
