function surrogate_NN_cfd_streamlined

%% SUMMARY =========================================================================================
% This code optimizes and trains a CNN on CFD data. This trained network relates the incoming wind
% speed and rotor speed to predict the thrust and torque on a 1.1 m residential turbine blade. This
% surrogate model will the be applied in artificial hybrid tests (run once per blade per time step)
% for quickly calculating aerodynamic blade loads based on the provided wind field during testing.
%
% This code is separated into the following sections:
%   1. Parameter input, which allows the users to specify all required parameters
%   2. Data importing, which imports the CFD data in tabular form
%   3. Data arrangement, which arranges the import data into the specific form required by
%       trainNetwork and splits it into training, validation and testing sets
%   4. Data pre-processing, which normalizes and shuffles data, divides into inputs and outputs 
%   5. Hyperparameter optimization, which performs two rounds of Bayesian optimization to
%       optimize the network's hyperparameters, the bounds of which are defined in Section 1
%   6. Final training, which trains the optimized network on the full set of data
%   7. Network testing, which checks both the accuracy of the network compared to the testing data
%       as well as quantifying the run time
%   8. Save results
%
% The CNN will relate 57 inputs per input time step to two outputs. These inputs are 33 along-wind
% wind speeds at 0.2R, 0.5R, and 1.0R upwind of the blade where R = 1.1 m = blade length; 11
% across-wind and 11 vertical wind speeds at 0.2R; the rotor speed, and the base 
% rotation. These 57 data points will be input per time step input into the CNN (the exact number 
% of which are a hyperparameter to optimize). The two outputs are the root thrust (along-wind force 
% (N)) and torque (moment about rotor axis (Nm)) for the given time step.
%
% Data will be input as [t, Ux_0.2R, Uy_0.2R, Uz_0.2R, Ux_0.5R, Ux_1.0R, w, theta, T, Q] where t 
% is time, w is the rotor speed, T is the thrust, Q is the torque. This data represents the time 
% history of a single blade during a single simulation, thus each CFD simulation yielded three such 
% time histories per wind speed/rotor speed configuration. Wind speeds are in m/s, rotor speed in 
% rad/s, blade number is 1, 2 or 3, theta is in rad, thrust in N and torque in N/m.



clear
clc
close all
%#ok<*NOPRT>

% Save output of command window to a log file
diary 'E:\PhD\Thesis\TestingTHs\NN Results\NN log'


%% 1. PARAMETER INPUT ==============================================================================

% Filepaths
filepath_data='E:\PhD\Thesis\TestingTHs\CFD Results\Processed Results\';
filepath_save='E:\PhD\Thesis\TestingTHs\NN Results\';

% DATA CONFIGURATION: use 62 inputs or 14 inputs? Original code was designed to run with 62 inputs,
% but has been updated to allow for fewer. For 64 inputs col_remove=0; for 14 inputs col_remove=48
col_remove=48; % number of leftmost columns to remove from input

% Static hyperparameter specification (conservative estimates made)
n_bayopt_evals1=60; % Number of Bayesian optimization evaluations in first round
n_bayopt_evals2=40; % Number of Bayesian optimization evaluations in second round
n_epochs_bayopt=50; % Number of epochs during optimization
n_epochs_final=5000; % Number of epochs during final training
init_learn_rate=0.001; % Initial learning rate
momentum=0.98; % Momentum

% Optimizable hyperparameter specifications
n_hyppar=6; % Number of hyperparameters to investigate (must equal number of rows in 
            % hyppar_table). hyppar = hyperparameters
hyppar_names={'ConvolutionFilterSize'; 'NumberOfFilters'; ...
    'NumberOfInputTimeSteps'; 'PoolingSize'; 'FirstFCLayerSize'; 'SecondFCLayerSize'}; 
    % FC = fully-connected
hyppar_range_low= [  1;  1;  11;  1;  10;   10];
hyppar_range_high=[ 10; 10;  30;  6; 1000; 1000]; % Initial high and low values for the respective 
                                              % hyperparameter, optimal value will be found between
                                              % these
hyppar_countstyle={'integer'; 'integer'; 'integer'; 'integer'; 'integer'; 'integer'}; 
    % Hyperparameters can count uniformly, by integer, and logarithmically. Correct choice depends 
    % on type of hyperparameter. ALl here are integers, but for example learning rate would be 
    % counted logarithmically while momentum would use the default
hyppar_secondround=[2; 4; 4; 2; 50; 50]; % Added/subtracted from optimal hyperparameter from round 1
    % to determine ranges in round 2

hyppar_table=table(hyppar_names, hyppar_range_low, hyppar_range_high, ...
    hyppar_countstyle, hyppar_secondround);
    % Table collecting hyperparameter names and ranges. Each row corresponds to one hyperparameter


    
%% 2. DATA IMPORTING ===============================================================================

% Data files
n_blades=3; % Number of turbine blades
n_wind=3; % Number of wind speed/TSR configurations [3.63/0.95, 5.15/1.08, 5.15/4.98]
    wind_names={'363_095'; '515_108'; '515_498'}; % Wind names for file calling
n_angles=8; % Number of base rotation angles [-0.01, 0.00, 0.01, 0.05, 0.10 ,0.15, 0.20, 0.25]
    angle_names={'N001'; 'P000'; 'P001'; 'P005'; 'P010'; 'P015'; 'P020'; 'P025'}; % Angle names
n_data_files_total=n_blades*n_wind*n_angles;
data_file_names=cell(n_data_files_total,1); % Preallocate cell size to assign data file names

% Create file names
for i=1:n_blades
    for j=1:n_wind
        for k=1:n_angles
            % The name of each data file
            data_file_names{(n_angles*n_wind)*(i-1)+n_angles*(j-1)+k,1}=...
                ['cfdresult_',angle_names{k,1},'_',wind_names{j,1},'_blade',num2str(i),'.csv']; 
        end
    end
end



% Import first data file
temp_data=readtable([filepath_data,data_file_names{1,1}]);
temp_data=table2array(temp_data); % Convert data imported as table into a matrix
[~,n_col]=size(temp_data); % Count the number of columns
n_col=n_col-1; % Reduce by 1 as the time column will be removed from the saved data

% Extract time information
t=temp_data(:,1); % Time history (s)
dt=t(2)-t(1); % Time step (2)
n_time=length(t); % Number of time steps
T=t(end,1); % Largest time step (s)

% File for data storage (3D matrix: rows=time, columns=input, depth=test case)
raw_data=zeros(n_time,n_col,n_data_files_total);

% Assign first data set to raw data matrix
raw_data(:,:,1)=temp_data(:,2:end);

% Assign remaining data
for i=2:n_data_files_total
    temp_data=readtable([filepath_data,data_file_names{i,1}]);
    temp_data=table2array(temp_data);
    raw_data(:,:,i)=temp_data(:,2:end);
end
clear temp_data

% Remove unwanted data based on col_remove
n_col=n_col-col_remove;
raw_data=raw_data(:,(col_remove+1):end,:);



%% 3. DATA ARRANGEMENT =============================================================================

% Normalize data to a possible range of [-1,1] and record the normalization factor (important). Data
% is normalized according to the training data but is applied to all data
norm_factors=zeros(n_col,1);
for i=1:n_col
   maximum=max(raw_data(:,i,:),[],'all');
   minimum=min(raw_data(:,i,:),[],'all');
   norm_max=max(maximum,abs(minimum));
   raw_data(:,i,:)=raw_data(:,i,:)/(norm_max);
   norm_factors(i)=1/(norm_max);
end


% Split data into Training (60%), Validation (20%), and Testing (20%) segments
% Since time step adjacency must be maintain, each TH will be split twice into 3 THs. This splitting
% is randomly done in one of six ways. Let Tr be training, V be validation, and Te be testing. For
% each TH, a random integer between 1 and 6 is chosen, and the TH is cut appropriately:
%   If rand = 1, the TH is cut into Tr,  V, Te
%   If rand = 2, the TH is cut into Tr, Te,  V
%   If rand = 3, the TH is cut into  V, Tr, Te
%   If rand = 4, the TH is cut into  V, Te, Tr
%   If rand = 5, the TH is cut into Te, Tr,  V
%   If rand = 6, the TH is cut into Te,  V, Tr
% The data histories (the stacking of the sheets in the 3D matrix) are then randomly shuffled

% Determine sizes of each split time history
n_test=floor(0.2*n_time); % Number of time steps in the testing segments
n_val=n_test; % Number of time steps in the validation segments
n_train=n_time-2*n_test; % Number of time steps in the training segments

% Preallocate matrix sizes
data_train=zeros(n_train,n_col,n_data_files_total);
data_val=zeros(n_val,n_col,n_data_files_total);
data_test=zeros(n_test,n_col,n_data_files_total);

% Split up full data samples into training, validation, testing
for i=1:n_data_files_total
    rand_num=randi(6); % Generate random integer to decide how data is split
    
    if rand_num==1
        data_train(:,:,i)=raw_data(1:n_train,:,i); % Split the time histories into three sections
        data_val(:,:,i)=raw_data(n_train+(1:n_val),:,i);
        data_test(:,:,i)=raw_data(n_train+n_val+(1:n_test),:,i); 
    
    elseif rand_num==2
        data_train(:,:,i)=raw_data(1:n_train,:,i);
        data_test(:,:,i)=raw_data(n_train+(1:n_test),:,i);
        data_val(:,:,i)=raw_data(n_train+n_test+(1:n_val),:,i);
    
    elseif rand_num==3
        data_val(:,:,i)=raw_data(1:n_val,:,i);
        data_train(:,:,i)=raw_data(n_val+(1:n_train),:,i);
        data_test(:,:,i)=raw_data(n_val+n_train+(1:n_test),:,i);  
    
    elseif rand_num==4
        data_val(:,:,i)=raw_data(1:n_val,:,i);
        data_test(:,:,i)=raw_data(n_val+(1:n_test),:,i);
        data_train(:,:,i)=raw_data(n_val+n_test+(1:n_train),:,i);    
        
    elseif rand_num==5
        data_test(:,:,i)=raw_data(1:n_test,:,i);
        data_train(:,:,i)=raw_data(n_test+(1:n_train),:,i);
        data_val(:,:,i)=raw_data(n_test+n_train+(1:n_val),:,i);
        
    elseif rand_num==6
        data_test(:,:,i)=raw_data(1:n_test,:,i);
        data_val(:,:,i)=raw_data(n_test+(1:n_val),:,i);
        data_train(:,:,i)=raw_data(n_test+n_val+(1:n_train),:,i);
    end
end
clear raw_data


% Shuffle 3D matrices: 3rd dimension is randomized, done independently for each data group
rand_order=randperm(n_data_files_total); % Generates random arrangement of integers from 1 to 
                                         % n_data_files_total
data_train=data_train(:,:,rand_order); % Randomize the stacking of the matrices in the training data
rand_order=randperm(n_data_files_total);
data_val=data_val(:,:,rand_order);
rand_order=randperm(n_data_files_total);
data_test=data_test(:,:,rand_order);


% Prepare to split into input and output data. Last two values are outputs (T and Q), rest are
% inputs
n_inputs=n_col-2;
n_outputs=2;


% Define function to rearrange data into the form required by the image input layer in trainNetwork
% Can't be performed ahead of time due to window size being an optimizable hyperparameter, therefore
% will be run inside of optimization/training loops
function [input_train, output_train, input_val, output_val, input_test, output_test]=...
        data_rearranger(data_train, data_val, data_test, windowsize)
    
    % Initialize matrices
    % Input into CNN as a number of 3D images (x,y,channels). Fourth dimension is the number of
    % images, which is equal to n_time-windowsize (as can't test until enough timesteps have passed
    % that a full window can be filled) per data sample
    n_train_images=(n_train-windowsize+1)*n_data_files_total;
    n_val_images=(n_val-windowsize+1)*n_data_files_total;
    n_test_images=(n_test-windowsize+1)*n_data_files_total;
    input_train=zeros(1, windowsize, n_inputs, n_train_images);
    input_val=zeros(1, windowsize, n_inputs, n_val_images);
    input_test=zeros(1, windowsize, n_inputs, n_test_images);
    
    % Outputs are 2D matrix equal to number of images x number of outputs
    output_train=zeros(n_train_images, n_outputs);
    output_val=zeros(n_val_images, n_outputs);
    output_test=zeros(n_test_images, n_outputs);

    
    % Assign training data
    for ii=1:n_data_files_total
        for jj=windowsize:n_train
            % Rearrangement is complex, but in broad strokes use reshape command to change 
                % time X input matrix into 1 X input X time matrix
            input_train(:,:,:,(n_train-windowsize+1)*(ii-1)+(jj+1-windowsize))=...
                reshape((data_train((jj-windowsize+1):jj,1:n_inputs,ii)),1,[],n_inputs);
            
            output_train((n_train-windowsize+1)*(ii-1)+(jj+1-windowsize),:)=...
                data_train(jj,(n_inputs+1):end,ii);
        end
    end
    
    % Shuffle training data so that adjacent time steps aren't next to each other in fourth
    % dimension, while making sure that inputs and outputs remain aligned
    rand_order=randperm(n_train_images); % Generate random order
    input_train=input_train(:,:,:,rand_order); % Shuffle along fourth dimension
    output_train=output_train(rand_order,:); % Shuffle along first dimension
    
    
    % Assign validation data
    for ii=1:n_data_files_total
        for jj=windowsize:n_val
            % Rearrangement is complex, but in broad strokes use reshape command to change 
                % time X input matrix into 1 X input X time matrix
            input_val(:,:,:,(n_val-windowsize+1)*(ii-1)+(jj+1-windowsize))=...
                reshape((data_val((jj-windowsize+1):jj,1:n_inputs,ii)),1,[],n_inputs);
            
            output_val((n_val-windowsize+1)*(ii-1)+(jj+1-windowsize),:)=...
                data_val(jj,(n_inputs+1):end,ii);
        end
    end
    
    % Shuffle validation data so that adjacent time steps aren't next to each other in fourth
    % dimension, while making sure that inputs and outputs remain aligned
    rand_order=randperm(n_val_images); % Generate random order
    input_val=input_val(:,:,:,rand_order); % Shuffle along fourth dimension
    output_val=output_val(rand_order,:); % Shuffle along first dimension
    
    
    % Assign testing data
    for ii=1:n_data_files_total
        for jj=windowsize:n_test
            % Rearrangement is complex, but in broad strokes use reshape command to change 
                % time X input matrix into 1 X input X time matrix
            input_test(:,:,:,(n_test-windowsize+1)*(ii-1)+(jj+1-windowsize))=...
                reshape((data_test((jj-windowsize+1):jj,1:n_inputs,ii)),1,[],n_inputs);
            
            output_test((n_test-windowsize+1)*(ii-1)+(jj+1-windowsize),:)=...
                data_test(jj,(n_inputs+1):end,ii);
        end
    end
    
    % Shuffle testing data so that adjacent time steps aren't next to each other in fourth
    % dimension, while making sure that inputs and outputs remain aligned
    rand_order=randperm(n_test_images); % Generate random order
    input_test=input_test(:,:,:,rand_order); % Shuffle along fourth dimension
    output_test=output_test(rand_order,:); % Shuffle along first dimension
end



%% 4. HYPERPARAMETER OPTIMIZATION ==================================================================

% Hyperparameter optimization is performed using Bayesian optimization. Firstly a function that
% trains a reduced network using a given set of hyperparameter values and evaluates the training must
% be developed. Then the Bayesian optimization program is run to estimate optimal parameter.
% Finally, the parameter ranges are reduced by the predicted optimal hyperparameters and a second
% round of Bayesian optimization is performed, where the predicted optimal hyperparameters are
% carried forward into the final simulation.

% Initialize neural network outside of a function
net = feedforwardnet(10);
% Initialize tracker, which helps track the progress of the program
tracker=0;

% Function for small set of training during hyperparameter optimization
function hyppar_opt_RMSE = hyppar_opt(parameters)
    
    % bayesopt function imports parameters as a table, convert it to a matrix
    parameters=table2array(parameters);
    cov_filt_size=parameters(1);
    n_filt=parameters(2);
    windowsize=parameters(3);
    pool_size=parameters(4);
    fully_con_size1=parameters(5);
    fully_con_size2=parameters(6);
    
    % The following value is the maximum possible size of the polling process (it pools the entire 
    % data set into a linear layer). If requested pool size exceeds this, must be reduced
    max_pool_size=windowsize+1-cov_filt_size;
    if max_pool_size<pool_size
       pool_size=max_pool_size; 
    end
    
    % Arrange data based on windowsize using pre-defined program, testing data only needed for
    % after the final training
    [input_train, output_train, input_val, output_val, ~, ~]=...
        data_rearranger(data_train, data_val, data_test, windowsize);
    
    % Increment tracker and print progress
    tracker=tracker+1;
    ['Hyperparameter optimization: iteration ',num2str(tracker),...
        ' of ', num2str(n_bayopt_evals1+n_bayopt_evals2),' (',num2str(n_bayopt_evals1),...
        ' in first round)'] 
    
    % Define the layers of the CNN
    layers = [ ...
        imageInputLayer([1,windowsize,n_inputs])
        convolution2dLayer([1,cov_filt_size],n_filt,'NumChannels',n_inputs)
        batchNormalizationLayer % Recommended by documentation
        tanhLayer
        averagePooling2dLayer([1,pool_size],'Stride',[1,1])           
        fullyConnectedLayer(fully_con_size1)
        tanhLayer
        fullyConnectedLayer(fully_con_size2)
        tanhLayer
        fullyConnectedLayer(n_outputs)
        regressionLayer];
    
    % analyzeNetwork(layers) % Uncommented command will generate figure of neural network, useful 
        % for trouble shooting
    
        
    % Since optimization is not occurring on full data set, will only supply a limited section of the
    % (randomized) training and validation data. One sixth of training and one third of testing data
    % will be supplied
    n_train_images_hypopt=floor((n_train-windowsize+1)*n_data_files_total/6);
    n_val_images_hypopt=floor((n_val-windowsize+1)*n_data_files_total/3);
    
    % Define NN options
    options = trainingOptions('sgdm', ...
        'MaxEpochs', n_epochs_bayopt, ...
        'InitialLearnRate',init_learn_rate, ...
        'Verbose',true, ...
        'VerboseFrequency',1000, ...
        'Momentum', momentum, ...
        'Shuffle','never', ...
        'LearnRateSchedule','piecewise', ...
        'MiniBatchSize', 10, ...
        'Plots','none', ...
        'ValidationData', ...
            {input_val(:,:,:,1:n_val_images_hypopt), output_val(1:n_val_images_hypopt,:)}, ...
        'ValidationPatience', 5);
        
    % Train small net
    net = trainNetwork(input_train(:,:,:,1:n_train_images_hypopt),...
        output_train(1:n_train_images_hypopt,:),layers,options);

    % Predict output of net
    output_val_predicted = predict(net,input_val(:,:,:,1:n_val_images_hypopt));
    output_val_true=output_val(1:n_val_images_hypopt,:);

    % Calculate RMSE of net based on predicted versus true validation data outputs
    hyppar_opt_RMSE=(((output_val_true-output_val_predicted)./output_val_true).^2);
    hyppar_opt_RMSE=sqrt(sum(hyppar_opt_RMSE,'all')/numel(hyppar_opt_RMSE));
%     hyppar_opt_RMSE=((output_val_true-output_val_predicted).^2);
%     hyppar_opt_RMSE=sqrt(sum(hyppar_opt_RMSE,'all')/numel(hyppar_opt_RMSE));
end


% Assign previously-declared variables to optimize in first round of hyperparameter
optimVars = [
    optimizableVariable(char(hyppar_table{1,1}),[hyppar_table{1,2},hyppar_table{1,3}],...
        'Type',char(hyppar_table{1,4}))
    optimizableVariable(char(hyppar_table{2,1}),[hyppar_table{2,2},hyppar_table{2,3}],...
        'Type',char(hyppar_table{2,4}))
    optimizableVariable(char(hyppar_table{3,1}),[hyppar_table{3,2},hyppar_table{3,3}],...
        'Type',char(hyppar_table{3,4}))
    optimizableVariable(char(hyppar_table{4,1}),[hyppar_table{4,2},hyppar_table{4,3}],...
        'Type',char(hyppar_table{4,4}))
    optimizableVariable(char(hyppar_table{5,1}),[hyppar_table{5,2},hyppar_table{5,3}],...
        'Type',char(hyppar_table{5,4}))
    optimizableVariable(char(hyppar_table{6,1}),[hyppar_table{6,2},hyppar_table{6,3}],...
        'Type',char(hyppar_table{6,4}))];


% Run first round of hyperparameter optimization and track required time
tic
optimal_hyperparameters = bayesopt(@hyppar_opt,optimVars,'Verbose',1000,...
    'MaxObjectiveEvaluations',n_bayopt_evals1,'UseParallel',false,'PlotFcn',[]);
toc

% Print out predicted hyperparameters
optimal_hyperparameters.XAtMinObjective


% Update hyperparameter ranges for second round of testing, the ranges are selected as the minimum/
% maxmimum of the base starting values and the optimal value from round 1 plus/minus the second 
% round range
optimVars = [
    optimizableVariable(char(hyppar_table{1,1}),...
        [max([hyppar_table{1,2},optimal_hyperparameters.XAtMinObjective{1,1}-hyppar_table{1,5}]),...
        min([hyppar_table{1,3},optimal_hyperparameters.XAtMinObjective{1,1}+hyppar_table{1,5}])],...
        'Type',char(hyppar_table{1,4}))
    optimizableVariable(char(hyppar_table{2,1}),...
        [max([hyppar_table{2,2},optimal_hyperparameters.XAtMinObjective{1,2}-hyppar_table{2,5}]),...
        min([hyppar_table{2,3},optimal_hyperparameters.XAtMinObjective{1,2}+hyppar_table{2,5}])],...
        'Type',char(hyppar_table{2,4}))
    optimizableVariable(char(hyppar_table{3,1}),...
        [max([hyppar_table{3,2},optimal_hyperparameters.XAtMinObjective{1,3}-hyppar_table{3,5}]),...
        min([hyppar_table{3,3},optimal_hyperparameters.XAtMinObjective{1,3}+hyppar_table{3,5}])],...
        'Type',char(hyppar_table{3,4}))
    optimizableVariable(char(hyppar_table{4,1}),...
        [max([hyppar_table{4,2},optimal_hyperparameters.XAtMinObjective{1,4}-hyppar_table{4,5}]),...
        min([hyppar_table{4,3},optimal_hyperparameters.XAtMinObjective{1,4}+hyppar_table{4,5}])],...
        'Type',char(hyppar_table{4,4}))
    optimizableVariable(char(hyppar_table{5,1}),...
        [max([hyppar_table{5,2},optimal_hyperparameters.XAtMinObjective{1,5}-hyppar_table{5,5}]),...
        min([hyppar_table{5,3},optimal_hyperparameters.XAtMinObjective{1,5}+hyppar_table{5,5}])],...
        'Type',char(hyppar_table{5,4}))
    optimizableVariable(char(hyppar_table{6,1}),...
        [max([hyppar_table{6,2},optimal_hyperparameters.XAtMinObjective{1,6}-hyppar_table{6,5}]),...
        min([hyppar_table{6,3},optimal_hyperparameters.XAtMinObjective{1,6}+hyppar_table{6,5}])],...
        'Type',char(hyppar_table{6,4}))];

% Run second round of hyperparameter optimization and track required time
tic
optimal_hyperparameters = bayesopt(@hyppar_opt,optimVars,'Verbose',1000,...
    'MaxObjectiveEvaluations',n_bayopt_evals2,'UseParallel',false,'PlotFcn',[]);
toc

% Print out optimal hyperparameters
optimal_hyperparameters.XAtMinObjective
    


%% 5. FINAL TRAINING ===============================================================================

% Assign optimized hyperparameters
parameters=table2array(optimal_hyperparameters.XAtMinObjective);
cov_filt_size=parameters(1);
n_filt=parameters(2);
windowsize=parameters(3);
pool_size=parameters(4);
fully_con_size1=parameters(5);
fully_con_size2=parameters(6);

% Reduce pool size if required
max_pool_size=windowsize+1-cov_filt_size;
if max_pool_size<pool_size
   pool_size=max_pool_size; 
end

% Arrange data based on windowsize using pre-defined program
[input_train, output_train, input_val, output_val, input_test, output_test]=...
    data_rearranger(data_train, data_val, data_test, windowsize);

% Define the layers of the CNN
layers = [ ...
    imageInputLayer([1,windowsize,n_inputs])
    convolution2dLayer([1,cov_filt_size],n_filt,'NumChannels',n_inputs)
    batchNormalizationLayer % Recommended by documentation
    tanhLayer
    averagePooling2dLayer([1,pool_size],'Stride',[1,1])           
    fullyConnectedLayer(fully_con_size1)
    tanhLayer
    fullyConnectedLayer(fully_con_size2)
    tanhLayer
    fullyConnectedLayer(n_outputs)
    regressionLayer];

% Define NN options
options = trainingOptions('sgdm', ...
    'MaxEpochs', n_epochs_final, ...
    'InitialLearnRate',init_learn_rate, ...
    'Verbose',true, ...
    'VerboseFrequency',1000, ...
    'Momentum', momentum, ...
    'Shuffle','never', ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize', 10, ...
    'Plots','none', ...
    'ValidationData', {input_val, output_val}, ...
    'ValidationPatience', 5);

% Train the final network
net = trainNetwork(input_train,output_train,layers,options);



%% 6. NETWORK TESTING ==============================================================================

% With network trained, the testing data must be fed into it and the predicted output compared to
% the true testing output data. The RMSE and NRMSE is calculated for both thrust and torque
% separately and as a whole

% Predict the output of the trained network when testing data offered as input:
output_test_predicted = predict(net,input_test);

% Preallocate RMSE matrix. Row 1 is RMSE, row 2 is NRMSE, columns are equal to n_outputs. Row 3 is
% the mean NRMSE across all outputs, so there is only one value in this row
RMSE_results=zeros(3,n_outputs);

% Calculate RMSE for each column of output data between predicted and true
for i=1:n_outputs
    % Get RMSE
%     RMSE_calc_temp=((output_test(:,i)-output_test_predicted(:,i))./output_test(:,i)).^2;
%     RMSE_results(1,i)=sqrt(sum(RMSE_calc_temp,'all')/numel(RMSE_calc_temp));
    RMSE_calc_temp=(output_test(:,i)-output_test_predicted(:,i)).^2;
    RMSE_results(1,i)=sqrt(sum(RMSE_calc_temp,'all')/numel(RMSE_calc_temp));
    
    % Get NRMSE
    RMSE_results(2,i)=RMSE_results(1,i)/...
        (max(output_test(:,i),[],'all')-min(output_test(:,i),[],'all'));
    
    % Un-normalize RMSE
    RMSE_results(1,i)=RMSE_results(1,i)/norm_factors(n_col-n_outputs+i);
end

% Find mean NRMSE
RMSE_results(3,1)=mean(RMSE_results(2,:));
% RMSE_results(3,2)=mean(RMSE_results(1,:));%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


% Check testing speed. Program is run six times on single image inputs, first is discarded and other
% five are used to determine the average speed of a single step
for i=1:11
    tic
    temp = predict(net,input_test(:,:,:,i));
    toc
end

RMSE_results


%% 7. SAVE RESULTS =================================================================================

% Optimal hyperparameter output is unnecessarily detailed, only need some specific information
optimal_hyperparameters_reduced={optimal_hyperparameters.MinObjective; ...
    optimal_hyperparameters.XAtMinObjective; ...
    optimal_hyperparameters.MinEstimatedObjective; ...
    optimal_hyperparameters.XAtMinEstimatedObjective; ...
    optimal_hyperparameters.NumObjectiveEvaluations; ...
    optimal_hyperparameters.TotalElapsedTime; ...
    optimal_hyperparameters.ObjectiveMinimumTrace; ...
    optimal_hyperparameters.EstimatedObjectiveMinimumTrace};

% Save the trained neural network with the normalization factors
save([filepath_save,'trained NN.mat'],'net','norm_factors','optimal_hyperparameters_reduced');

% Save most of the results (input and output data will not be saved)
save([filepath_save,'full results.mat'],'net','optimal_hyperparameters_reduced','RMSE_results',...
    'norm_factors','output_test','output_test_predicted','input_test');

'PROGRAM COMPLETE'

% End command window logging
diary off

end