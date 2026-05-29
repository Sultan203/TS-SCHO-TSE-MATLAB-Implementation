TS-SCHO-TSE MATLAB Implementation
=================================

This package provides a reproducible MATLAB-style implementation of the TS-SCHO-TSE concept:
- TS-SCHO: Two-Stage Enhanced Sinh-Cosh Optimizer
- TSE: Two-Stage adaptive weighted ensemble

How to run:
1. Put your dataset CSV in the same folder or provide its full path.
2. Open run_TSSCHO_TSE_demo.m.
3. Set:
      dataPath = "your_dataset.csv";
      labelColumn = "";   % leave empty if the class label is the last column
4. Run the script.

Expected dataset format:
- Rows: network flow samples
- Columns: numeric/categorical features + one label column
- Last column is label by default

Important:
- This code is built from the methodology described in the paper and is intended for
  reproducibility support. It is not claimed to be the authors' original source code.
- For full paper-style experiments, set:
      opts.numRuns = 30;
      opts.populationSize = 50;
      opts.maxIterations = 200;
- Runtime can be high because each candidate solution is evaluated using 5-fold CV.

Files:
- run_TSSCHO_TSE_demo.m       Main runner
- loadDDoSDataset.m           CSV loader
- ts_scho_tse.m               Optimizer
- fitness_tse.m               Fitness function
- decode_solution.m           Candidate decoder
- stratifiedKFoldEval.m       Cross-validation evaluation
- train_tse_ensemble.m        Train base learners
- predict_tse_ensemble.m      Weighted voting prediction
- classificationMetrics.m     Metrics

Required MATLAB toolboxes:
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox is optional for fitcnet. If unavailable, the code uses a fallback classifier.
