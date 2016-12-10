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

package org.deidentifier.arx.criteria;

import java.io.Serializable;

import org.deidentifier.arx.ARXPopulationModel;
import org.deidentifier.arx.DataSubset;
import org.deidentifier.arx.framework.check.groupify.HashGroupifyEntry;
import org.deidentifier.arx.framework.data.DataManager;

/**
 * An abstract base class for privacy criteria.
 *
 * @author Fabian Prasser
 * @author Florian Kohlmayer
 */
public abstract class PrivacyCriterion implements Serializable{

    /**  SVUID */
    private static final long serialVersionUID = -8460571120677880409L;
    
    /** Is the criterion monotonic with generalization and suppression. */
    private final boolean monotonic;
    

    /** Is the criterion monotonic with generalization. */
    private final Boolean monotonicWithGeneralization;
    
    /**
     * Instantiates a new criterion.
     *
     * @param monotonicWithSuppression
     * @param monotonicWithGeneralization
     */
    public PrivacyCriterion(boolean monotonicWithSuppression,
                            boolean monotonicWithGeneralization){
        this.monotonic = monotonicWithSuppression;
        this.monotonicWithGeneralization = monotonicWithGeneralization;
    }
    
    /**
     * Clone
     */
    public abstract PrivacyCriterion clone();
    
    /**
     * Returns the associated population model, <code>null</code> if there is none.
     * @return the populationModel
     */
    public ARXPopulationModel getPopulationModel() {
        return null;
    }

    /**
     * Returns the criterion's requirements.
     *
     * @return
     */
    public abstract int getRequirements();
    
    /**
     * Return journalist risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdJournalist() {
        return 1d;
    }

    /**
     * Return marketer risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdMarketer() {
        return 1d;
    }
    
    /**
     * Return prosecutor risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdProsecutor() {
        return 1d;
    }
    
    /**
     * Override this to initialize the criterion.
     *
     * @param manager
     */
    public void initialize(DataManager manager){
        // Empty by design
    }
    
    /**
     * Returns a research subset, <code>null</code> if no subset is available
     * @return
     */
    public DataSubset getSubset() {
        return null;
    }
    
    /**
     * Implement this, to enforce the criterion.
     *
     * @param entry
     * @return
     */
    public abstract boolean isAnonymous(HashGroupifyEntry entry);
    
    /**
     * Returns whether the criterion supports local recoding.
     * @return
     */
    public abstract boolean isLocalRecodingSupported();
    
    /**
     * Returns whether the criterion is monotonic with generalization.
     * @return
     */
    public boolean isMonotonicWithGeneralization() {
        // Default
        if (this.monotonicWithGeneralization == null) {
            return true;
        } else {
            return this.monotonicWithGeneralization;
        }
    }

    /**
     * Returns whether the criterion is monotonic with tuple suppression.
     *
     * @return
     */
    public boolean isMonotonicWithSuppression() {
        return this.monotonic;
    }

    /**
     * Is this criterion based on the overall sample
     * @return
     */
    public boolean isSampleBased() {
        return false;
    }

    /**
     * Returns a string representation.
     *
     * @return
     */
    public abstract String toString();
}
