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

package org.deidentifier.arx.framework.check.transformer;

import org.deidentifier.arx.ARXConfiguration.ARXConfigurationInternal;
import org.deidentifier.arx.framework.check.distribution.IntArrayDictionary;
import org.deidentifier.arx.framework.data.GeneralizationHierarchy;

/**
 * The class Transformer12.
 * 
 * @author Fabian Prasser
 * @author Florian Kohlmayer
 */
public class Transformer12 extends AbstractTransformer {

    /**
     * Instantiates a new transformer.
     *
     * @param data the data
     * @param hierarchies the hierarchies
     * @param otherValues
     * @param dictionarySensValue
     * @param dictionarySensFreq
     * @param config
     */
    public Transformer12(final int[][] data,
                         final GeneralizationHierarchy[] hierarchies,
                         final int[][] otherValues,
                         final IntArrayDictionary dictionarySensValue,
                         final IntArrayDictionary dictionarySensFreq,
                         final ARXConfigurationInternal config) {
        super(data, hierarchies, otherValues, dictionarySensValue, dictionarySensFreq, config);
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.deidentifier.ARX.framework.check.transformer.AbstractTransformer
     * #walkAll()
     */
    @Override
    protected void processAll() {
        for (int i = startIndex; i < stopIndex; i++) {
            intuple = data[i];
            outtuple = buffer[i];
            outtuple[outindex0] = idindex0[intuple[index0]][generalizationindex0];
            outtuple[outindex1] = idindex1[intuple[index1]][generalizationindex1];
            outtuple[outindex2] = idindex2[intuple[index2]][generalizationindex2];
            outtuple[outindex3] = idindex3[intuple[index3]][generalizationindex3];
            outtuple[outindex4] = idindex4[intuple[index4]][generalizationindex4];
            outtuple[outindex5] = idindex5[intuple[index5]][generalizationindex5];
            outtuple[outindex6] = idindex6[intuple[index6]][generalizationindex6];
            outtuple[outindex7] = idindex7[intuple[index7]][generalizationindex7];
            outtuple[outindex8] = idindex8[intuple[index8]][generalizationindex8];
            outtuple[outindex9] = idindex9[intuple[index9]][generalizationindex9];
            outtuple[outindex10] = idindex10[intuple[index10]][generalizationindex10];
            outtuple[outindex11] = idindex11[intuple[index11]][generalizationindex11];

            // Call
            delegate.callAll(outtuple, i);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.deidentifier.ARX.framework.check.transformer.AbstractTransformer
     * #walkGroupify ()
     */
    @Override
    protected void processGroupify() {

        while (element != null) {

            intuple = data[element.representative];
            outtuple = buffer[element.representative];
            outtuple[outindex0] = idindex0[intuple[index0]][generalizationindex0];
            outtuple[outindex1] = idindex1[intuple[index1]][generalizationindex1];
            outtuple[outindex2] = idindex2[intuple[index2]][generalizationindex2];
            outtuple[outindex3] = idindex3[intuple[index3]][generalizationindex3];
            outtuple[outindex4] = idindex4[intuple[index4]][generalizationindex4];
            outtuple[outindex5] = idindex5[intuple[index5]][generalizationindex5];
            outtuple[outindex6] = idindex6[intuple[index6]][generalizationindex6];
            outtuple[outindex7] = idindex7[intuple[index7]][generalizationindex7];
            outtuple[outindex8] = idindex8[intuple[index8]][generalizationindex8];
            outtuple[outindex9] = idindex9[intuple[index9]][generalizationindex9];
            outtuple[outindex10] = idindex10[intuple[index10]][generalizationindex10];
            outtuple[outindex11] = idindex11[intuple[index11]][generalizationindex11];

            // Call
            delegate.callGroupify(outtuple, element);

            // Next element
            element = element.nextOrdered;
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.deidentifier.ARX.framework.check.transformer.AbstractTransformer
     * #walkSnapshot ()
     */
    @Override
    protected void processSnapshot() {
        startIndex *= ssStepWidth;
        stopIndex *= ssStepWidth;

        for (int i = startIndex; i < stopIndex; i += ssStepWidth) {
            intuple = data[snapshot[i]];
            outtuple = buffer[snapshot[i]];
            outtuple[outindex0] = idindex0[intuple[index0]][generalizationindex0];
            outtuple[outindex1] = idindex1[intuple[index1]][generalizationindex1];
            outtuple[outindex2] = idindex2[intuple[index2]][generalizationindex2];
            outtuple[outindex3] = idindex3[intuple[index3]][generalizationindex3];
            outtuple[outindex4] = idindex4[intuple[index4]][generalizationindex4];
            outtuple[outindex5] = idindex5[intuple[index5]][generalizationindex5];
            outtuple[outindex6] = idindex6[intuple[index6]][generalizationindex6];
            outtuple[outindex7] = idindex7[intuple[index7]][generalizationindex7];
            outtuple[outindex8] = idindex8[intuple[index8]][generalizationindex8];
            outtuple[outindex9] = idindex9[intuple[index9]][generalizationindex9];
            outtuple[outindex10] = idindex10[intuple[index10]][generalizationindex10];
            outtuple[outindex11] = idindex11[intuple[index11]][generalizationindex11];

            // Call
            delegate.callSnapshot(outtuple, snapshot, i);
        }
    }
}
