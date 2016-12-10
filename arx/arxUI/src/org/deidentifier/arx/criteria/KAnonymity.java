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

import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.framework.check.groupify.HashGroupifyEntry;

/**
 * The k-anonymity criterion
 * Published in:
 * Sweeney L. 
 * k-anonymity: A model for protecting privacy. 
 * International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems. 2002;10(5):557 - 570. 
 * 
 * @author Fabian Prasser
 * @author Florian Kohlmayer
 */
public class KAnonymity extends ImplicitPrivacyCriterion{

    /**  SVUID */
    private static final long serialVersionUID = -8370928677928140572L;
    
    /** The parameter k. */
    private final int k;
    
    /**
     * Creates a new instance of the k-anonymity criterion as proposed in
     * Sweeney L. k-Anonymity: A model for protecting privacy. 
     * International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems. 2002;10(5):557 - 570. 
     * @param k
     */
    public KAnonymity(int k){
        super(true, true);
        this.k = k;
    }

    @Override
    public KAnonymity clone() {
        return new KAnonymity(this.getK());
    }

    /**
     * Returns the parameter k.
     *
     * @return
     */
    public int getK() {
        return k;
    }

    @Override
    public int getRequirements(){
        // Requires only one counter
        return ARXConfiguration.REQUIREMENT_COUNTER;
    }
    
	@Override
    public boolean isAnonymous(HashGroupifyEntry entry) {
        throw new RuntimeException("This should never be called!");
    }

    @Override
    public boolean isLocalRecodingSupported() {
        return true;
    }

    @Override
	public String toString() {
		return k+"-anonymity";
	}

    /**
     * Return prosecutor risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdProsecutor() {
        return 1d / (double)k;
    }

    /**
     * Return journalist risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdJournalist() {
        return getRiskThresholdProsecutor();
    }

    /**
     * Return marketer risk threshold, 1 if there is none
     * @return
     */
    public double getRiskThresholdMarketer() {
        return getRiskThresholdProsecutor();
    }
}