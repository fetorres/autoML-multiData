/**
 * Added by Yunhui
 * Save the output file and privacy criterions for AutoML 
 */

package org.deidentifier.arx.gui.worker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Set;

import org.apache.commons.io.output.CountingOutputStream;
import org.deidentifier.arx.ARXPopulationModel;
import org.deidentifier.arx.ARXPopulationModel.Region;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.criteria.PrivacyCriterion;
import org.deidentifier.arx.gui.Controller;
import org.deidentifier.arx.gui.model.Model;
import org.deidentifier.arx.gui.resources.Resources;
import org.deidentifier.arx.io.CSVSyntax;
import org.eclipse.core.runtime.IProgressMonitor;

public class WorkerAutoMLSaver extends Worker<Model> {
	/** Output specification files */
	private final String outSpec;
	/** Number of saved privacy configurations */
	/** output file name for data */
	private final String dataName;
	/** output configuration name for data */
	private final String configName;

	/** Controller */
	private final Controller controller;
	/** Model */
	private final Model model;

	/** The stop flag. */
	private volatile boolean stop = false;

	public WorkerAutoMLSaver(final Controller controller, final Model model, final String outPath, final String outSpec,
			final int savedCnt) {
		this.outSpec = outSpec;
		this.model = model;
		this.controller = controller;
		dataName = outPath + savedCnt + ".csv";
		configName = outPath + savedCnt + "_config.txt";
	}

	public void run(final IProgressMonitor arg0) throws InvocationTargetException, InterruptedException {

		arg0.beginTask(Resources.getMessage("WorkerExport.0"), 100); //$NON-NLS-1$

		try {
			updateOutputSpec();
			savePrivConfig();
		} catch (Exception e) {
			error = e;
			e.printStackTrace();
			arg0.done();
			return;
		}

		// Create output stream
		final File file = new File(dataName);
		FileOutputStream out = null;
		try {
			out = new FileOutputStream(file);
		} catch (final FileNotFoundException e) {
			error = e;
			arg0.done();
			return;
		}

		double bytes = model.getInputBytes();
		// Track progress
		final CountingOutputStream count = new CountingOutputStream(out);
		final Thread t = new Thread(new Runnable() {
			@Override
			public void run() {
				int previous = 0;
				while ((count.getByteCount() != bytes) && !stop) {
					int progress = (int) ((double) count.getByteCount() / (double) bytes * 100d);
					if (progress != previous) {
						arg0.worked(progress - previous);
						previous = progress;
					}
					try {
						Thread.sleep(100);
					} catch (final InterruptedException e) {
						/* Ignore */}
				}
			}
		});
		t.setDaemon(true);
		t.start();

		// Export the data
		try {
			DataHandle dh = model.getOutput();
			CSVSyntax config = new CSVSyntax(',');
			dh.save(dataName, config);
			stop = true;
			arg0.done();
		} catch (final Exception e) {
			error = e;
			stop = true;
			arg0.done();
			return;
		}
	}

	/**
	 * append new file names to output.txt
	 * 
	 * @throws IOException
	 */
	private void updateOutputSpec() throws IOException {
		// append new file names to output.txt
		BufferedWriter w0 = new BufferedWriter(new FileWriter(outSpec, true));
		controller.increaseSavedCnt();

		w0.write(dataName);
		w0.newLine();
		// Append the privacy configuration files (don't need this for now)
		// w0.write(configName);
		// w0.newLine();

		w0.flush();
		w0.close();
	}

	/**
	 * write privacy criterion to [savedCnt]_config.txt
	 * 
	 * @throws IOException
	 */

	private void savePrivConfig() throws IOException {
		// write privacy criterion to i_config.txt
		BufferedWriter wConfig = new BufferedWriter(new FileWriter(configName));

		System.out.println("Privacy criterion saved: ");
		Set<PrivacyCriterion> pcSet = model.getInputConfig().getConfig().getCriteria();
		for (PrivacyCriterion pc : pcSet) {
			wConfig.write("Privacy Criterion: " + pc.toString());
			System.out.println("Privacy Criterion: " + pc.toString());
			wConfig.newLine();
		}
		// write risks to i_config.txt
		double riskAv = model.getOutput().getRiskEstimator(ARXPopulationModel.create(Region.USA))
				.getSampleBasedReidentificationRisk().getAverageRisk();
			wConfig.newLine();
		if (Double.isNaN(riskAv)) {
			riskAv = 0.0;
		}
		wConfig.write("Average Re-identification Risk: " + riskAv);
		wConfig.newLine(); 
//		System.out.println("Average Re-identification Risk: " + riskAv);

		if (riskAv > 0) {
			double riskHigh = model.getOutput().getRiskEstimator(ARXPopulationModel.create(Region.USA))
					.getSampleBasedReidentificationRisk().getHighestRisk();
			wConfig.write("Highest Re-identification Risk: " + riskHigh);
			wConfig.newLine();
//			System.out.println("Highest Re-identification Risk: " + riskHigh);
			wConfig.newLine();
		}
		System.out.println();
		wConfig.flush();
		wConfig.close();
	}

}
