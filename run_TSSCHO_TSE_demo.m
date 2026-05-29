

clear; clc; close all;

dataPath = "dataset.csv";      
labelColumn = "";                  
positiveClass = "";                

opts.populationSize = 50;
opts.maxIterations = 200;
opts.numFolds = 5;
opts.numRuns = 1;                  
opts.eliteRatio = 0.20;
opts.randomSeed = 42;

% Fitness weights: minimize fitness
opts.wAcc = 0.70;                  
opts.wF1  = 0.20;                  
opts.wFeat = 0.07;                 
opts.wCost = 0.03;                 

% Classifiers in the ensemble
opts.learners = ["SVM","RF","KNN","NB","MLP"];

%% ---------------- Load data ----------------
if ~isfile(dataPath)
    error("Please set dataPath to your CSV dataset file.");
end

[X, y, featureNames] = loadDDoSDataset(dataPath, labelColumn);

% Convert labels to categorical
y = categorical(y);

%% ---------------- Run TS-SCHO-TSE ----------------
results = ts_scho_tse(X, y, opts);

disp("Best result:");
disp(results.bestMetrics);
fprintf("Selected features: %d / %d\n", sum(results.bestSolution.featureMask), size(X,2));
disp("Selected feature names:");
disp(featureNames(results.bestSolution.featureMask)');

%% ---------------- Plot convergence ----------------
figure;
plot(results.convergence, 'LineWidth', 1.8);
xlabel('Iteration');
ylabel('Best Fitness');
title('TS-SCHO-TSE Convergence Curve');
grid on;
