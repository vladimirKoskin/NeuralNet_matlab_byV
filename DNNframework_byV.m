% THE ULTIMATE NEURAL NETWORK OPTIMZATION by Vlad
clear all

%% Hyperparameters

    %%% Architecture
NbClasses = 2; % Generally the same as the dim of last layer but not necessarily

%%%% GOOD ONES:
LayerDimensions = [4,3,2]; % From input layer to output layer (SHORTEST/SMALLEST to perfect work for artificial data)
LayerActivations = [0,1,2];

% LayerDimensions = [4,2,4,2]; % From input layer to output layer
% LayerActivations = [0,1,1,2];

% LayerDimensions = [4,3,2,3,2]; % From input layer to output layer (SMALLEST BOTTLENECK (2-node layer here) for perfect work for artificial data)
% LayerActivations = [0,1,1,1,2];
%%%%%%%%%%%
% LayerActivations::: 0 : No Activation / 1 : ReLU / 2 : Softmax / (3 : ELU to implement later)

    %%% Initialisation
wInitCoeff = 0.1; %(generally 1/order of max layer in base 10 is good, actually making different wInitCoeff foer different dimensions is good)

    %%% Learning
learnRate = 0.2;
NbEpochs = 2000; 


%% Initialisation

NbLayers = length(LayerDimensions);

[LayerList,WeightList,BiasList] = deal({});

for i = 1:NbLayers
    LayerList{i} = zeros(LayerDimensions(i),1);
end

for i = 1:NbLayers-1
   WeightList{i} =  (rand(LayerDimensions(i),LayerDimensions(i+1)) -0.5)*wInitCoeff; % Random Initial Weights
   BiasList{i} = ones(LayerDimensions(i+1),1);
end
% WeightList{1}; % Check

%% Data Pipeline
disp('START Loading Data ...')


%% Artificial Data INIT
TrainDataClass = {};
TestDataClass = {};

% insert Artificial data for maintenance (CONVEX vs CONCAVE)
a = [1,2,3];
TrainDataClass{1} = [[2.5,5.2,3.2,1.8];[1,4,6,1.2];[1,2,7,3];[2,5,8,6]]';
TrainDataClass{2} = [[4,3.1,3,4];[4.2,3.2,4,6];[5,4,5,7];[5,3,2,4]]';

TestDataClass{1} = [[1,3,5,2];[3.5,5,4.1,3.1];[2,4,3,2.8]]';
TestDataClass{2} = [[3.8,2.1,1,5.1];[5,3,3.2,4];[5,4.1,2.3,4.8]]';
%

TrainDataClass{1,2}(:,1);
figure; hold on
for k = 1:length(TrainDataClass{2}(1,:))
subplot(2,1,1); hold on
    plot(TrainDataClass{1}(:,k),'b-') % '-' TRAIN DATA
    plot(TrainDataClass{2}(:,k),'r-')
    title('Artificial Train Data')
end

for k = 1:length(TestDataClass{2}(1,:))
subplot(2,1,2); hold on
    plot(TestDataClass{1}(:,k),'b--','LineWidth',3) % '--' means TEST DATA
    plot(TestDataClass{2}(:,k),'r--','LineWidth',3)
    title('Artificial Test Data')
end
%% LEARNING
disp('START Learning ...')

OutputCount = zeros(1,LayerDimensions(length(LayerDimensions)));
for u = 1:NbEpochs

% Initialise empty weights & bias gradients
WeightGradientList = {};
BiasGradientList = {};
for i = 1:NbLayers-1
   WeightGradientList{i} =  zeros(LayerDimensions(i),LayerDimensions(i+1));
   BiasGradientList{i} = zeros(LayerDimensions(i+1),1);
end 


% Pick random samples (all classes with (almost) equiprobability)
nbSample = 1; % Not used actually

randPickedClass = fix(rand(1)*NbClasses)+1; % Pre-pick the class/target with uniform probability
randPickedIndex = fix(rand(1)*length(TrainDataClass{randPickedClass}(1,:)))+1; % Pick an index withing the pre-picked class
pickedDat = TrainDataClass{randPickedClass}(:,randPickedIndex); % Get the data tuple from the pre-picked class as the training sample


trueIndx = randPickedClass;
target = zeros(LayerDimensions(NbLayers),1); 
target(trueIndx) = 1; % Make 1-hot/1-k encoding vector for the target

% FORWARD PROPAGATION
LayerList{1}  = pickedDat' ; % Insert the picked data tuple into the Input Layer
dActList = {}; % Stock of derivatives of activation functions

for afk = 2:NbLayers
    
%%% Forw Propagation
    LayerList{afk} = LayerList{afk-1}*WeightList{afk-1} + BiasList{afk-1}';
   
%%% Activation
    if LayerActivations(afk) == 1 % ReLU
        LayerList{afk} = arrayfun(@(x) max(x,0),LayerList{afk}); % ReLU Activation    
    elseif LayerActivations(afk) == 2 % SoftMax
        LayerList{afk} = arrayfun(@(x) exp(x),LayerList{afk}); % Softmax Activation 1/2
        LayerList{afk} = LayerList{afk} ./sum(LayerList{afk}); % Softmax Activation 2/2     
    else 
                'Problem with activation functions parameters'
    end
end

% Compute Activation Derivatives
dActList{1} = ones(1,LayerDimensions(1)); % Stock of derivatives of activation functions
for afk = 2:NbLayers
    if LayerActivations(afk) == 1 % ReLU        
        dActList{afk} = ones(1,LayerDimensions(afk)); % ReLU derivative 1/2
        dActList{afk}(LayerList{afk} == 0) = 0; % ReLU Derivative 2/2       
    elseif LayerActivations(afk) == 2 % SoftMax       
        dActList{afk} = ones(1,LayerDimensions(afk)); % Assuming that we minimize the Cross-entropy function
    else 
                'Problem with activation functions parameters'
    end
end

% Recording stuff
OutputCount(trueIndx) = OutputCount(trueIndx) + 1;
OutputRecord(OutputCount(trueIndx),:,trueIndx) = LayerList{NbLayers};

 
%%%% Backpropagation
% Get the derivatives
dErr = (LayerList{NbLayers}' - target); % dE/dy 
[maxVal maxIndx] = max(abs(LayerList{NbLayers}));

if  max(abs(dErr)) > 0.01 %% Backpropagate only if wrong
% maxIndx ~= trueIndx ||
       
% Bias and Weight Gradients
theBest = sort(1:NbLayers-2,'descend');
BiasGradientList{NbLayers-1} = dErr.*dActList{NbLayers}' ;
WeightGradientList{NbLayers-1} = (BiasGradientList{NbLayers-1}*LayerList{NbLayers-1})';
for kcl = theBest      
    BiasGradientList{kcl} = WeightList{kcl+1}*BiasGradientList{kcl+1}.*dActList{kcl+1}' ;
    WeightGradientList{kcl} = (BiasGradientList{kcl}*LayerList{kcl})';
end

for kcl = theBest % WEIGHTS UPDATE
    WeightList{kcl} =WeightList{kcl} -learnRate*WeightGradientList{kcl};
    BiasList{kcl} = BiasList{kcl} - learnRate*BiasGradientList{kcl};
end

end % END {if (wrong) then backpropagate}
end % END random iteration over train dataset


%% Check Learning
figure; hold on
subplot(2,1,1); hold on
    title('Predicted values for TRAIN Samples in B')
    plot(OutputRecord(:,2,1),'r.')
    plot(OutputRecord(:,1,1),'g.')
    % xlim([100000 250000])
    legend('P(B|B)','P(A|B)');
    xlabel('Training Epochs')
subplot(2,1,2); hold on
    title('Predicted values for TRAIN Samples in A')
    plot(OutputRecord(:,2,2),'g.')
    plot(OutputRecord(:,1,2),'r.')
    % xlim([100000 250000])
    legend('P(A|A)','P(B|A)');
    xlabel('Training Epochs')



%% TEST
disp('START Testing ...')

record_output = zeros(2,3,2); % Records the output of last layer, per neuron per sample per class.
record_prediction = zeros(3,2); % Records the prediction (index of max value of last layer), per sample per class

for class_id = 1:2
    classwise_test_data = TestDataClass{class_id};
    
for sample_id = 1:size(TestDataClass{class_id},2)
    
    % FORWARD PROPAGATION
    LayerList{1}  = classwise_test_data(:,sample_id)' ; % Insert the picked data tuple into the Input Layer
    dActList = {}; % Stock of derivatives of activation functions

    for afk = 2:NbLayers

    %%% Forw Propagation
        LayerList{afk} = LayerList{afk-1}*WeightList{afk-1} + BiasList{afk-1}';

    %%% Activation
        if LayerActivations(afk) == 1 % ReLU
            LayerList{afk} = arrayfun(@(x) max(x,0),LayerList{afk}); % ReLU Activation    
        elseif LayerActivations(afk) == 2 % SoftMax
            LayerList{afk} = arrayfun(@(x) exp(x),LayerList{afk}); % Softmax Activation 1/2
            LayerList{afk} = LayerList{afk} ./sum(LayerList{afk}); % Softmax Activation 2/2     
        else 
                    'Problem with activation functions parameters'
        end
    end
    record_output(:,sample_id,class_id) = LayerList{NbLayers};
    [maxVal,maxPos] = max(LayerList{NbLayers});
    record_prediction(sample_id,class_id) = maxPos;
end
end

%% Check Test Samples (Confusion Matrix)

confusion_mat = zeros(NbClasses);

for true_class_id = 1:NbClasses
    for prediction_id = 1:length(record_prediction(:,true_class_id))
        predicted_class = record_prediction(prediction_id,true_class_id);
        confusion_mat(true_class_id,predicted_class) = confusion_mat(true_class_id,predicted_class) + 1;
    end
    confusion_mat(true_class_id,:) = confusion_mat(true_class_id,:)./sum(confusion_mat(true_class_id,:)); % Normalize the line
end
figure; hold on
    title('Confusion Matrix of our Model on the Test Dataset')
    imagesc(confusion_mat)

    x = repmat(1:NbClasses,NbClasses,1); % generate x-coordinates
    y = x'; % generate y-coordinates
    t = cellfun(@num2str, num2cell(confusion_mat), 'UniformOutput', false); % convert to string
    text(x(:), y(:), t, 'HorizontalAlignment', 'Center','FontSize',20)
    
    xlabel('True State')
    xticks([1,2])
    xticklabels({'B','A'})
    yticks([1,2])
    yticklabels({'B','A'})
    ylabel('Predicted State') % Maybe x/y axis are messed up here could not bother checking

