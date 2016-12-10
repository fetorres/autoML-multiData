/*
 * ARX: Powerful Data Anonymization
 * Copyright 2012 - 2016 Fabian Prasser, Florian Kohlmayer and contributors
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.deidentifier.arx.aggregates;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deidentifier.arx.ARXLogisticRegressionConfiguration;
import org.deidentifier.arx.DataHandleInternal;
import org.deidentifier.arx.aggregates.classification.ClassificationDataSpecification;
import org.deidentifier.arx.aggregates.classification.ClassificationMethod;
import org.deidentifier.arx.aggregates.classification.ClassificationResult;
import org.deidentifier.arx.aggregates.classification.MultiClassLogisticRegression;
import org.deidentifier.arx.aggregates.classification.MultiClassZeroR;
import org.deidentifier.arx.common.WrappedBoolean;
import org.deidentifier.arx.exceptions.ComputationInterruptedException;

/**
 * Statistics representing the prediction accuracy of a data mining
 * classification operator
 * 
 * @author Fabian Prasser
 */
public class StatisticsClassification {
    
    /**
     * A matrix mapping confidence thresholds to precision and recall
     * 
     * @author Fabian Prasser
     *
     */
    public static class PrecisionRecallMatrix {
        
        /** Confidence thresholds*/
        private static final double[] CONFIDENCE_THRESHOLDS = new double[]{
            0d, 0.1d, 0.2d, 0.3d, 0.4d, 0.5d, 0.6d, 0.7d, 0.8d, 0.9d, 1d
        };

        /** Measurements */
        private double                measurements          = 0d;
        /** Precision */
        private final double[]        precision             = new double[CONFIDENCE_THRESHOLDS.length];
        /** Recall */
        private final double[]        recall                = new double[CONFIDENCE_THRESHOLDS.length];

        /**
         * @return the confidence thresholds
         */
        public double[] getConfidenceThresholds() {
            return CONFIDENCE_THRESHOLDS;
        }
        
        /**
         * @return the precision
         */
        public double[] getPrecision() {
            return precision;
        }

        /**
         * @return the recall
         */
        public double[] getRecall() {
            return recall;
        }

        /**
         * Adds a new value
         * @param confidence
         * @param correct
         */
        void add(double confidence, boolean correct) {
            
            for (int i = 0; i < CONFIDENCE_THRESHOLDS.length; i++) {
                if (confidence >= CONFIDENCE_THRESHOLDS[i]) {
                    recall[i]++;
                    precision[i] += correct ? 1d : 0d;
                }
            }
            measurements++;
        }

        /**
         * Packs the results
         */
        void pack() {
            // Pack
            for (int i = 0; i < CONFIDENCE_THRESHOLDS.length; i++) {
                
                if (recall[i] == 0d) {
                    precision[i] = 1d;
                } else {
                    precision[i] /= recall[i];
                    recall[i] /= measurements;
                }
            }
        }
    }

    /** Accuracy */
    private double                accuracy;
    /** Average error */
    private double                averageError;
    /** Interrupt flag */
    private final WrappedBoolean  interrupt;
    /** Precision/recall matrix */
    private PrecisionRecallMatrix matrix         = new PrecisionRecallMatrix();
    /** Num classes */
    private int                   numClasses;
    /** Original accuracy */
    private double                originalAccuracy;
    /** Original accuracy */
    private double                originalAverageError;
    /** Precision/recall matrix */
    private PrecisionRecallMatrix originalMatrix = new PrecisionRecallMatrix();
    /** Random */
    private final Random          random;
    /** ZeroR accuracy */
    private double                zeroRAccuracy;
    /** ZeroR accuracy */
    private double                zeroRAverageError;
    /** Measurements */
    private int                   numMeasurements;

    /**
     * Creates a new set of statistics for the given classification task
     * @param inputHandle - The input features handle
     * @param outputHandle - The output features handle
     * @param features - The feature attributes
     * @param clazz - The class attributes
     * @param config - The configuration
     * @param interrupt - The interrupt flag
     * @throws ParseException 
     */
    StatisticsClassification(DataHandleInternal inputHandle,
                             DataHandleInternal outputHandle,
                             String[] features,
                             String clazz,
                             ARXLogisticRegressionConfiguration config,
                             WrappedBoolean interrupt) throws ParseException {

        // Init
        this.interrupt = interrupt;
        
        // Check and clean up
        double samplingFraction = (double)config.getMaxRecords() / (double)inputHandle.getNumRows();
        if (samplingFraction <= 0d) {
            throw new IllegalArgumentException("Sampling fraction must be >0");
        }
        if (samplingFraction > 1d) {
            samplingFraction = 1d;
        }
        
       
        // Initialize random
        if (!config.isDeterministic()) {
            this.random = new Random();
        } else {
            this.random = new Random(config.getSeed());
        }
        
        // TODO: Feature is not used. Continuous variables are treated as categorical.
        ClassificationDataSpecification specification = new ClassificationDataSpecification(inputHandle, 
                                                                                            outputHandle, 
                                                                                            features,
                                                                                            clazz,
                                                                                            interrupt);
        
        // Train and evaluate
        int k = inputHandle.getNumRows() > config.getNumFolds() ? config.getNumFolds() : inputHandle.getNumRows();
        List<List<Integer>> folds = getFolds(inputHandle.getNumRows(), k);

        // Track
        int classifications = 0;
        
        // For each fold as a validation set
        for (int evaluationFold = 0; evaluationFold < folds.size(); evaluationFold++) {
            
            // Create classifiers
            ClassificationMethod inputLR = new MultiClassLogisticRegression(specification, config);
            ClassificationMethod inputZR = new MultiClassZeroR(specification);
            ClassificationMethod outputLR = null;
            if (inputHandle != outputHandle) {
                outputLR = new MultiClassLogisticRegression(specification, config);
            }
            
            // Try
            try {
                
                // Train with all training sets
                boolean trained = false;
                for (int trainingFold = 0; trainingFold < folds.size(); trainingFold++) {
                    if (trainingFold != evaluationFold) {                        
                        for (int index : folds.get(trainingFold)) {
                            checkInterrupt();
                            inputLR.train(inputHandle, outputHandle, index);
                            inputZR.train(inputHandle, outputHandle, index);
                            if (outputLR != null && !outputHandle.isOutlier(index)) {
                                outputLR.train(outputHandle, outputHandle, index);
                            }
                            trained = true;
                        }
                    }
                }
                
                // Close
                inputLR.close();
                inputZR.close();
                if (outputLR != null) {
                    outputLR.close();
                }
                
                // Now validate
                for (int index : folds.get(evaluationFold)) {
                    
                    // Check
                    checkInterrupt();
                    
                    // If trained
                    if (trained) {
                        
                        // Classify
                        ClassificationResult resultInputLR = inputLR.classify(inputHandle, index);
                        ClassificationResult resultInputZR = inputZR.classify(inputHandle, index);
                        ClassificationResult resultOutputLR = outputLR == null ? null : outputLR.classify(outputHandle, index);
                        classifications++;
                        
                        // Correct result
                        String actualValue = outputHandle.getValue(index, specification.classIndex, true);
                        
                        // Maintain data about inputZR
                        this.zeroRAverageError += resultInputZR.error(actualValue);
                        this.zeroRAccuracy += resultInputZR.correct(actualValue) ? 1d : 0d;

                        // Maintain data about inputLR
                        boolean correct = resultInputLR.correct(actualValue);
                        this.originalAverageError += resultInputLR.error(actualValue);
                        this.originalAccuracy += correct ? 1d : 0d;
                        this.originalMatrix.add(resultInputLR.confidence(), correct);

                        // Maintain data about outputLR                        
                        if (resultOutputLR != null) {
                            correct = resultOutputLR.correct(actualValue);
                            this.averageError += resultOutputLR.error(actualValue);
                            this.accuracy += correct ? 1d : 0d;
                            this.matrix.add(resultOutputLR.confidence(), correct);
                        }
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        
        
        // Maintain data about inputZR
        this.zeroRAverageError /= (double)classifications;
        this.zeroRAccuracy/= (double)classifications;

        // Maintain data about inputLR
        this.originalAverageError /= (double)classifications;
        this.originalAccuracy /= (double)classifications;
        this.originalMatrix.pack();

        // Maintain data about outputLR                        
        if (inputHandle != outputHandle) {
            this.averageError /= (double)classifications;
            this.accuracy /= (double)classifications;
            this.matrix.pack();
        } else {
            this.averageError = this.originalAverageError;
            this.accuracy = this.originalAccuracy;
            this.matrix = this.originalMatrix;
        }
        
        this.numClasses = specification.classMap.size();
        this.numMeasurements = classifications;
    }

    /**
     * Returns the resulting accuracy. Obtained by training a
     * Logistic Regression classifier on the output (or input) dataset.
     * 
     * @return
     */
    public double getAccuracy() {
        return this.accuracy;
    }
    
    /**
     * Returns the average error, defined as avg(1d-probability-of-correct-result) for
     * each classification event.
     * 
     * @return
     */
    public double getAverageError() {
        return this.averageError;
    }

    /**
     * Returns the number of classes
     * @return
     */
    public int getNumClasses() {
        return this.numClasses;
    }
    
    /**
     * Returns the number of measurements
     * @return
     */
    public int getNumMeasurements() {
        return this.numMeasurements;
    }
    
    /**
     * Returns the maximal accuracy. Obtained by training a
     * Logistic Regression classifier on the input dataset.
     * 
     * @return
     */
    public double getOriginalAccuracy() {
        return this.originalAccuracy;
    }

    /**
     * Returns the average error, defined as avg(1d-probability-of-correct-result) for
     * each classification event.
     * 
     * @return
     */
    public double getOriginalAverageError() {
        return this.originalAverageError;
    }

    /**
     * Returns a precision/recall matrix for LogisticRegression on input
     * @return
     */
    public PrecisionRecallMatrix getOriginalPrecisionRecall() {
        return this.originalMatrix;
    }
    
    /**
     * Returns a precision/recall matrix
     * @return
     */
    public PrecisionRecallMatrix getPrecisionRecall() {
        return this.matrix;
    }

    /**
     * Returns the minimal accuracy. Obtained by training a
     * ZeroR classifier on the input dataset.
     * 
     * @return
     */
    public double getZeroRAccuracy() {
        return this.zeroRAccuracy;
    }
    
    /**
     * Returns the average error, defined as avg(1d-probability-of-correct-result) for
     * each classification event.
     * 
     * @return
     */
    public double getZeroRAverageError() {
        return this.zeroRAverageError;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("StatisticsClassification{\n");
        builder.append(" - Accuracy:\n");
        builder.append("   * Original: ").append(originalAccuracy).append("\n");
        builder.append("   * ZeroR: ").append(zeroRAccuracy).append("\n");
        builder.append("   * Output: ").append(accuracy).append("\n");
        builder.append(" - Average error:\n");
        builder.append("   * Original: ").append(originalAverageError).append("\n");
        builder.append("   * ZeroR: ").append(zeroRAverageError).append("\n");
        builder.append("   * Output: ").append(averageError).append("\n");
        builder.append(" - Number of classes: ").append(numClasses).append("\n");
        builder.append(" - Number of measurements: ").append(numMeasurements).append("\n");
        builder.append("}");
        return builder.toString();
    }

    /**
     * Checks whether an interruption happened.
     */
    private void checkInterrupt() {
        if (interrupt.value) {
            throw new ComputationInterruptedException("Interrupted");
        }
    }
    
    /**
     * Creates the folds
     * @param length
     * @param k
     * @param random
     * @return
     */
    private List<List<Integer>> getFolds(int length, int k) {
        
        // Prepare indexes
        List<Integer> rows = new ArrayList<>();
        for (int row = 0; row < length; row++) {
            rows.add(row);
        }
        Collections.shuffle(rows, random);
        
        // Create folds
        List<List<Integer>> folds = new ArrayList<>();
        int size = rows.size() / k;
        size = size > 1 ? size : 1;
        for (int i = 0; i < k; i++) {
            // For each fold
            int min = i * size;
            int max = (i + 1) * size;
            if (i == k - 1) {
                max = rows.size();
            }

            // Collect rows
            List<Integer> fold = new ArrayList<>();
            for (int j = min; j < max; j++) {
                fold.add(rows.get(j));
            }

            // Store
            folds.add(fold);
        }

        // Free
        rows.clear();
        rows = null;
        return folds;
    }
}