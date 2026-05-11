function ds = downsample_calcium_and_spikes(calcium, spikes, original_fps, target_fps)
% downsample_calcium_and_spikes
%
% Downsample one calcium fluorescence trace and its matching binned spike train.
%
% Inputs:
%   calcium      - calcium fluorescence trace
%   spikes       - binned spike train, same length as calcium
%   original_fps - original sampling rate, usually around 100 Hz
%   target_fps   - requested target sampling rate, e.g. 100, 50, 20, 10
%
% Output:
%   ds - struct with downsampled data and sanity-check information
%
% Logic:
%   Calcium is averaged inside each new time bin.
%   Spikes are summed inside each new time bin.
%
% Why:
%   Calcium is a continuous signal.
%   Spikes are event counts, so the total spike count must be preserved
%   after trimming.

    %% ---------------- Basic input checks ----------------

    if nargin ~= 4
        error(['downsample_calcium_and_spikes requires exactly 4 inputs: ', ...
               'calcium, spikes, original_fps, target_fps.']);
    end

    if isempty(calcium)
        error('Input calcium trace is empty.');
    end

    if isempty(spikes)
        error('Input spike train is empty.');
    end

    if isempty(original_fps) || ~isnumeric(original_fps) || ~isscalar(original_fps)
        error('original_fps must be a single numeric value.');
    end

    if isempty(target_fps) || ~isnumeric(target_fps) || ~isscalar(target_fps)
        error('target_fps must be a single numeric value.');
    end

    if original_fps <= 0
        error('original_fps must be positive. Received: %.4f', original_fps);
    end

    if target_fps <= 0
        error('target_fps must be positive. Received: %.4f', target_fps);
    end

    % Allow small fps differences around nominal values.
    % Example: original_fps = 99.999 and target_fps = 100 should be treated as no downsampling.
    fps_tolerance = 1.0;  % Hz
    
    if target_fps > original_fps + fps_tolerance
        error(['target_fps cannot be meaningfully higher than original_fps. ', ...
               'Received target_fps = %.4f, original_fps = %.4f.'], ...
               target_fps, original_fps);
    end

    %% ---------------- Convert to column vectors ----------------

    % Convert both inputs to column vectors.
    % This avoids shape problems later during reshape().
    calcium = double(calcium(:));
    spikes = double(spikes(:));

    %% ---------------- Signal sanity checks ----------------

    if length(calcium) ~= length(spikes)
        error(['Calcium and spikes must have the same length. ', ...
               'Calcium length = %d, spikes length = %d.'], ...
               length(calcium), length(spikes));
    end

    if any(isnan(calcium))
        error('Calcium trace contains NaN values.');
    end

    if any(isinf(calcium))
        error('Calcium trace contains Inf values.');
    end

    if any(isnan(spikes))
        error('Spike train contains NaN values.');
    end

    if any(isinf(spikes))
        error('Spike train contains Inf values.');
    end

    if any(spikes < 0)
        error('Spike train contains negative values, which is invalid for spike counts.');
    end

    % Spikes should usually be integer counts.
    % We allow tiny floating-point differences, but not real non-integer values.
    if any(abs(spikes - round(spikes)) > 1e-10)
        error('Spike train contains non-integer values. Expected binned spike counts.');
    end

    %% ---------------- Downsampling factor ----------------

    % We currently support simple integer downsampling.
    % Example:
    %   100 Hz -> 50 Hz: factor 2
    %   100 Hz -> 20 Hz: factor 5
    %   100 Hz -> 10 Hz: factor 10
    % If the requested target rate is very close to the original rate,
    % treat it as no downsampling.
    if abs(target_fps - original_fps) <= fps_tolerance
        downsample_factor = 1;
    else
        downsample_factor = round(original_fps / target_fps);
    end

    if downsample_factor < 1
        error('Calculated downsample_factor is smaller than 1. Check original_fps and target_fps.');
    end

    actual_fps = original_fps / downsample_factor;

    % Check whether the requested target rate is close to the actual rate
    % produced by integer downsampling.
    fps_error = abs(actual_fps - target_fps);

    % Allow a larger tolerance for nominal sampling rates like 100 Hz,
    % because some recordings are stored as 99.999 or 100.0002 Hz.
    if fps_error > fps_tolerance
        error(['Requested target_fps cannot be represented well by integer downsampling. ', ...
               'Requested %.4f Hz, actual %.4f Hz, error %.4f Hz.'], ...
               target_fps, actual_fps, fps_error);
    end

    %% ---------------- Trim signals to complete blocks ----------------

    n_original_samples = length(calcium);
    n_blocks = floor(n_original_samples / downsample_factor);
    n_trimmed_samples = n_blocks * downsample_factor;
    n_removed_samples = n_original_samples - n_trimmed_samples;

    if n_blocks < 1
        error(['Recording is too short for this downsampling factor. ', ...
               'Samples = %d, downsample_factor = %d.'], ...
               n_original_samples, downsample_factor);
    end

    calcium_trimmed = calcium(1:n_trimmed_samples);
    spikes_trimmed = spikes(1:n_trimmed_samples);

    removed_spike_count = sum(spikes(n_trimmed_samples+1:end));

    %% ---------------- Reshape into blocks ----------------

    % Each column is one new lower-rate time bin.
    % Each column contains downsample_factor original samples.
    calcium_blocks = reshape(calcium_trimmed, downsample_factor, n_blocks);
    spike_blocks = reshape(spikes_trimmed, downsample_factor, n_blocks);

    %% ---------------- Downsample ----------------

    % Calcium: average within each new bin.
    calcium_ds = mean(calcium_blocks, 1)';

    % Spikes: sum within each new bin.
    % This preserves spike counts after trimming.
    spikes_ds = sum(spike_blocks, 1)';

    %% ---------------- Post-downsampling sanity checks ----------------

    % Total spike count:
    % This counts the actual number of spikes across all bins.
    original_spike_count = sum(spikes);
    trimmed_spike_count = sum(spikes_trimmed);
    downsampled_spike_count = sum(spikes_ds);
    
    % Active spike bins:
    % This counts how many time bins contain at least one spike.
    % This number may decrease after downsampling because several nearby spikes
    % can collapse into the same lower-rate bin.
    original_active_spike_bins = sum(spikes > 0);
    trimmed_active_spike_bins = sum(spikes_trimmed > 0);
    downsampled_active_spike_bins = sum(spikes_ds > 0);
    
    % Multi-spike bins:
    % These are bins that contain more than one spike.
    % They become more common at lower sampling rates.
    original_multi_spike_bins = sum(spikes > 1);
    trimmed_multi_spike_bins = sum(spikes_trimmed > 1);
    downsampled_multi_spike_bins = sum(spikes_ds > 1);
    
    % How many active bins were merged due to downsampling.
    % This is not spike loss. It means temporal resolution was reduced.
    active_bin_reduction = trimmed_active_spike_bins - downsampled_active_spike_bins;

    % The key check:
    % Downsampled spike count must exactly match the spike count after trimming.
    if trimmed_spike_count ~= downsampled_spike_count
        error(['Spike count was not preserved during downsampling. ', ...
               'Trimmed spike count = %.0f, downsampled spike count = %.0f.'], ...
               trimmed_spike_count, downsampled_spike_count);
    end

    if length(calcium_ds) ~= length(spikes_ds)
        error(['Downsampled calcium and spike vectors have different lengths. ', ...
               'Calcium length = %d, spikes length = %d.'], ...
               length(calcium_ds), length(spikes_ds));
    end

    if any(isnan(calcium_ds)) || any(isinf(calcium_ds))
        error('Downsampled calcium contains NaN or Inf values.');
    end

    if any(isnan(spikes_ds)) || any(isinf(spikes_ds))
        error('Downsampled spikes contain NaN or Inf values.');
    end

    if any(spikes_ds < 0)
        error('Downsampled spikes contain negative values, which should be impossible.');
    end

    %% ---------------- Package output ----------------

    ds = struct();

    % Downsampled data
    ds.calcium = calcium_ds;
    ds.spikes = spikes_ds;

    % Sampling information
    ds.original_fps = original_fps;
    ds.target_fps = target_fps;
    ds.actual_fps = actual_fps;
    ds.downsample_factor = downsample_factor;

    % Sample counts
    ds.n_original_samples = n_original_samples;
    ds.n_trimmed_samples = n_trimmed_samples;
    ds.n_removed_samples = n_removed_samples;
    ds.n_downsampled_samples = length(calcium_ds);

    % Spike-count checks
    ds.original_spike_count = original_spike_count;
    ds.trimmed_spike_count = trimmed_spike_count;
    ds.downsampled_spike_count = downsampled_spike_count;
    ds.removed_spike_count = removed_spike_count;

    % Duration checks
    ds.original_duration_sec = n_original_samples / original_fps;
    ds.trimmed_duration_sec = n_trimmed_samples / original_fps;
    ds.downsampled_duration_sec = length(calcium_ds) / actual_fps;

    % Spike-count checks
    ds.original_spike_count = original_spike_count;
    ds.trimmed_spike_count = trimmed_spike_count;
    ds.downsampled_spike_count = downsampled_spike_count;
    ds.removed_spike_count = removed_spike_count;
    
    % Active spike-bin checks
    ds.original_active_spike_bins = original_active_spike_bins;
    ds.trimmed_active_spike_bins = trimmed_active_spike_bins;
    ds.downsampled_active_spike_bins = downsampled_active_spike_bins;
    ds.active_bin_reduction = active_bin_reduction;
    
    % Multi-spike-bin checks
    ds.original_multi_spike_bins = original_multi_spike_bins;
    ds.trimmed_multi_spike_bins = trimmed_multi_spike_bins;
    ds.downsampled_multi_spike_bins = downsampled_multi_spike_bins;

    % General status flag
    ds.sanity_checks_passed = true;
end