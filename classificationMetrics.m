function metrics = classificationMetrics(yTrue, yPred, posScore)
%CLASSIFICATIONMETRICS Binary metrics. For multiclass, macro precision/recall/F1 approximation.

yTrue = categorical(yTrue);
yPred = categorical(yPred);
classes = categories(yTrue);

accuracy = mean(yTrue == yPred);

if numel(classes) == 2
    posClass = classes{2};
    TP = sum(yTrue == posClass & yPred == posClass);
    TN = sum(yTrue ~= posClass & yPred ~= posClass);
    FP = sum(yTrue ~= posClass & yPred == posClass);
    FN = sum(yTrue == posClass & yPred ~= posClass);

    precision = TP / max(TP + FP, eps);
    recall = TP / max(TP + FN, eps);
    f1 = 2 * precision * recall / max(precision + recall, eps);

    if nargin >= 3 && ~isempty(posScore)
        try
            [~,~,~,auc] = perfcurve(yTrue, posScore, posClass);
        catch
            auc = NaN;
        end
    else
        auc = NaN;
    end
else
    precisions = zeros(numel(classes),1);
    recalls = zeros(numel(classes),1);
    f1s = zeros(numel(classes),1);
    for c = 1:numel(classes)
        cls = classes{c};
        TP = sum(yTrue == cls & yPred == cls);
        FP = sum(yTrue ~= cls & yPred == cls);
        FN = sum(yTrue == cls & yPred ~= cls);
        precisions(c) = TP / max(TP+FP, eps);
        recalls(c) = TP / max(TP+FN, eps);
        f1s(c) = 2*precisions(c)*recalls(c) / max(precisions(c)+recalls(c), eps);
    end
    precision = mean(precisions);
    recall = mean(recalls);
    f1 = mean(f1s);
    auc = NaN;
end

metrics = struct();
metrics.Accuracy = accuracy;
metrics.Precision = precision;
metrics.Recall = recall;
metrics.F1 = f1;
metrics.AUC = auc;
end
