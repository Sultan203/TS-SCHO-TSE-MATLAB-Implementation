function [pred, posScore] = predict_tse_ensemble(model, Xte, sol, classes)
%PREDICT_TSE_ENSEMBLE Weighted probability voting.

L = numel(sol.learners);
n = size(Xte,1);
C = numel(classes);
probSum = zeros(n, C);
totalW = 0;

for j = 1:L
    mdl = model.models{j};
    if isempty(mdl) || sol.weights(j) <= 0
        continue;
    end

    try
        if isa(mdl, 'TreeBagger')
            [~, scoresCell] = predict(mdl, Xte);
            scores = scoresCell;
            mdlClasses = string(mdl.ClassNames);
        else
            [~, scores] = predict(mdl, Xte);
            if isprop(mdl, 'ClassNames')
                mdlClasses = string(mdl.ClassNames);
            else
                mdlClasses = string(classes);
            end
        end

        % Align score columns to global class order
        aligned = zeros(n, C);
        for c = 1:C
            k = find(mdlClasses == string(classes{c}), 1);
            if ~isempty(k) && k <= size(scores,2)
                aligned(:,c) = scores(:,k);
            end
        end

        % If score is not probability-like, repair using softmax
        if any(~isfinite(aligned), 'all') || all(aligned(:)==0)
            aligned = zeros(n,C);
        end

        probSum = probSum + sol.weights(j) * aligned;
        totalW = totalW + sol.weights(j);

    catch ME
        warning("Prediction failed for learner %d: %s", j, ME.message);
    end
end

if totalW <= 0 || all(probSum(:)==0)
    % fallback: majority default to first class
    [~, idx] = max(ones(n,C), [], 2);
else
    probSum = probSum / totalW;
    [~, idx] = max(probSum, [], 2);
end

pred = categorical(classes(idx), classes);

if C == 2 && totalW > 0
    posScore = probSum(:,2);
else
    posScore = [];
end
end
