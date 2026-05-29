function model = train_tse_ensemble(Xtr, ytr, sol)
%TRAIN_TSE_ENSEMBLE Train active learners.

learners = sol.learners;
model.models = cell(numel(learners),1);
model.classes = categories(ytr);

for j = 1:numel(learners)
    if ~sol.activeLearners(j)
        continue;
    end

    name = upper(string(learners(j)));

    try
        switch name
            case "SVM"
                model.models{j} = fitcsvm(Xtr, ytr, ...
                    'KernelFunction','rbf', ...
                    'BoxConstraint', sol.hp.svmC, ...
                    'KernelScale', 1/max(sol.hp.svmGamma, eps), ...
                    'Standardize', false, ...
                    'ClassNames', categorical(model.classes));
                try
                    model.models{j} = fitPosterior(model.models{j}, Xtr, ytr);
                catch
                end

            case "RF"
                model.models{j} = TreeBagger(sol.hp.rfTrees, Xtr, ytr, ...
                    'Method','classification', ...
                    'OOBPrediction','off');

            case "KNN"
                model.models{j} = fitcknn(Xtr, ytr, ...
                    'NumNeighbors', sol.hp.knnK, ...
                    'Standardize', false);

            case "NB"
                model.models{j} = fitcnb(Xtr, ytr);

            case "MLP"
                % fitcnet requires newer MATLAB versions.
                try
                    model.models{j} = fitcnet(Xtr, ytr, ...
                        'LayerSizes', sol.hp.mlpHidden, ...
                        'Standardize', false, ...
                        'Verbose', 0);
                catch
                    % Fallback if fitcnet is unavailable
                    t = templateTree('MaxNumSplits', 20);
                    model.models{j} = fitcensemble(Xtr, ytr, ...
                        'Method','Bag', ...
                        'Learners', t, ...
                        'NumLearningCycles', 30);
                end
        end
    catch ME
        warning("Failed to train %s: %s", name, ME.message);
        model.models{j} = [];
    end
end
end
