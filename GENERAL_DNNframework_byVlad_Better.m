clear all % THE ULTIMATE NEURAL NETWORK OPTIMZATION by Vlad
%% Hyperparameters
NbClasses = 2; % Generally the same as the dim of last layer but not necessarily
LayerDimensions = [4,3,2]; % From input layer to output layer
LayerActivations = [0,1,2];
% 0 : No Activation / 1 : ReLU / 2 : Softmax / (3 : ELU to implement later)
wInitCoeff = 0.1; %(generally 1/order of max layer in base 10 is good, actually making different wInitCoeff foer different dimensions is good)
learnRate = 0.2;

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
%% Data Pipeline

%% Artificial Data INIT

TrainDataClass = {};
TestDataClass = {};

% insert Artificial data for maintenance / debugging
a = [1,2,3];
TrainDataClass{1} = [[1,3,5,2];[1,4,6,1.2];[1,2,7,3];[2,5,8,6]]';
TrainDataClass{2} = [[3.8,2.1,1,5.1];[4.2,3.2,4,6];[5,4,5,7];[5,3,2,4]]';

TestDataClass{1} = [[1,2.2,3.2,2];[3.5,4.1,5,3.1];[2,3,4,2.8]]';
TestDataClass{2} = [[4,3,3.1,4];[5,3.2,3,4];[5,2.3,4.1,4.8]]';

TrainDataClass{1,2}(:,1);
for k = 1:length(TrainDataClass{2}(1,:))
    figure(555); hold on
    plot(TrainDataClass{1}(:,k),'b-') % '-' TRAIN DATA
    plot(TrainDataClass{2}(:,k),'r-')
    title('Artificial Train Data')
end

for k = 1:length(TestDataClass{2}(1,:))
    figure(555); hold on
    plot(TestDataClass{1}(:,k),'b--','LineWidth',3) % '--' means TEST DATA
    plot(TestDataClass{2}(:,k),'r--','LineWidth',3)
    title('Artificial Test Data')
end
%% LEARNING
NbEpochs = 500;
OutputCount = zeros(1,LayerDimensions(length(LayerDimensions)));
for u = 1:NbEpochs
% Initialise empty weights & bias gradients
[WeightGradientList,BiasGradientList] = deal({});
for i = 1:NbLayers-1
   WeightGradientList{i} =  zeros(LayerDimensions(i),LayerDimensions(i+1));
   BiasGradientList{i} = zeros(LayerDimensions(i+1),1);
end 

% Pick random samples (all classes with (almost) equiprobability)
nbSample = 1;

randPickedClass = fix(rand(1)*NbClasses)+1;
randPickedIndex = fix(rand(1)*length(TrainDataClass{randPickedClass}(1,:)))+1;
pickedDat = TrainDataClass{randPickedClass}(:,randPickedIndex);

target = zeros(LayerDimensions(NbLayers),1);
target(randPickedClass) = 1;
trueIndx = randPickedClass;

% FORWARD PROPAGATION
LayerList{1}  = pickedDat' ;
dActList = {}; % Stock of derivatives of activation functions

for afk = 2:NbLayers
    LayerList{afk} = LayerList{afk-1}*WeightList{afk-1} + BiasList{afk-1}';
   
    % Activation
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


figure(5); hold on
plot(OutputRecord(:,2,1),'r.')
plot(OutputRecord(:,1,1),'g.')
% xlim([100000 250000])
legend('P(Rest|Rest)','P(Act|Rest)');

figure(6); hold on
plot(OutputRecord(:,2,2),'g.')
plot(OutputRecord(:,1,2),'r.')

% xlim([100000 250000])
legend('P(Act|Act)','P(Rest|Act)');
