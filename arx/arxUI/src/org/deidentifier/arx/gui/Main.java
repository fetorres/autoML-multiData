package org.deidentifier.arx.gui;

import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;

import javax.swing.JOptionPane;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.deidentifier.arx.gui.resources.Resources;
import org.deidentifier.arx.gui.view.impl.MainSplash;
import org.deidentifier.arx.gui.view.impl.MainWindow;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Monitor;

public class Main {

	/** Is the project already loaded. */
	private static String loaded = null;

	/** The splash. */
	private static MainSplash splash = null;

	/** The main window. */
	private static MainWindow main = null;

	/**
	 * Main entry point.
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		main(null, args);
	}

	/**
	 * Main entry point.
	 *
	 * @param args
	 */
	public static void main(Display display, final String[] args) {

		try {
			// Make fall-back toolkit look like the native UI
			if (!isUnix()) { // See:
								// https://bugs.eclipse.org/bugs/show_bug.cgi?id=341799
				UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
			}
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException
				| UnsupportedLookAndFeelException e) {
			// Ignore
		}

		try {

			// Display
			if (display == null) {
				display = new Display();
			}

			// Monitor
			Monitor monitor = getMonitor(display);

			// Splash
			splash = new MainSplash(display, monitor);
			splash.show();

			// Create main window
			main = new MainWindow(display, monitor);

			// Handler for loading a project
			if (args.length > 0 && args[0].endsWith(".deid")) { //$NON-NLS-1$
				main.onShow(new Runnable() {
					public void run() {
						load(main, args[0]);
					}
				});
			}

			// added by Yunhui
			// Handler for creating a new project and loading a file
			StringBuilder sb = new StringBuilder();
			if (args.length > 1 && args[1].endsWith(".csv")) { //$NON-NLS-1$
				String temp = args[0];
				if (!temp.endsWith("/") && !temp.endsWith("\\")) {
					temp = temp + '/';
				}
				final String path = temp;
				File fpath = new File(path);
				File fdata = new File(args[1]);
				if (fpath.exists() && fpath.isDirectory() && fdata.exists()) { // only call the modified method if both path and data are provided and exist
					main.onShow(new Runnable() {
						public void run() {
							loadFile(main, path, args[1]);
						}
					});
				} 
				else {
					if (!fpath.exists()) {
						sb.append("Path " + path +" does not exists!\n"); //TODO add different error messages
					} if (!fpath.isDirectory()) {
						sb.append("Path " + path + " is not a directory!\n");
					} if (!fdata.exists()) {
						sb.append("The input file " + args[1] + " does not exist!\n");
					}
					System.out.println(sb.toString());
				}
			}

			// Show window
			main.show();

			new Update(main.getShell());
			
			// Added by Yunhui, show load data error message
			if(sb.toString().length() > 0) {
				main.showInfoDialog(main.getShell(), "Load Data Error", sb.toString());
			}
			
			// Main event loop
			while (!main.isDisposed()) {
				try {

					// Event handling
					if (!display.readAndDispatch()) {
						display.sleep();
					}
				} catch (final Exception e) {

					// Error handling
					main.showErrorDialog(Resources.getMessage("MainWindow.9") + Resources.getMessage("MainWindow.10"), //$NON-NLS-1$ //$NON-NLS-2$
							e);
					StringWriter sw = new StringWriter();
					PrintWriter pw = new PrintWriter(sw);
					e.printStackTrace(pw);
					main.getController().getResources().getLogger().info(sw.toString());
				}
			}

			// Dispose display
			if (!display.isDisposed()) {
				display.dispose();
			}
		} catch (Throwable e) {

			// Error handling outside of SWT
			if (splash != null)
				splash.hide();
			StringWriter sw = new StringWriter();
			PrintWriter pw = new PrintWriter(sw);
			e.printStackTrace(pw);
			final String trace = sw.toString();

			// Show message
			JOptionPane.showMessageDialog(null, trace, "Unexpected error", JOptionPane.ERROR_MESSAGE); //$NON-NLS-1$
			System.exit(1);

		}
	}

	/**
	 * Returns the monitor on which the application was launched.
	 *
	 * @param display
	 * @return
	 */
	private static Monitor getMonitor(Display display) {
		Point mouse = display.getCursorLocation();
		for (Monitor monitor : display.getMonitors()) {
			if (monitor.getBounds().contains(mouse)) {
				return monitor;
			}
		}
		return display.getPrimaryMonitor();
	}

	/**
	 * Loads a project.
	 *
	 * @param main
	 * @param path
	 */
	private static void load(MainWindow main, String path) {
		if (loaded == null) {
			loaded = path;
			if (splash != null)
				splash.hide();
			main.getController().actionOpenProject(path);
		}
	}

	/**
	 * Determine os
	 * 
	 * @return
	 */
	private static boolean isUnix() {
		String os = System.getProperty("os.name").toLowerCase(); //$NON-NLS-1$
		return (os.indexOf("nix") >= 0 || os.indexOf("nux") >= 0 || os.indexOf("aix") > 0); //$NON-NLS-1$ //$NON-NLS-2$ //$NON-NLS-3$
	}
	
/**
 * added by Yunhui
 * Determine OS of current system
 * @return
 */
	public static int getOS() {
		String os = System.getProperty("os.name").toLowerCase(); //$NON-NLS-1$
		if (os.indexOf("win") >= 0) {
			return 1; // windows
		} else if (os.indexOf("mac") >= 0) {
			return 2; // Mac OS
		} else {
			return 0; // Unix
		}
		
	}

	/**
	 * added by Yunhui create a new project and load the file
	 * 
	 * @param main
	 * @param path
	 */
	private static void loadFile(MainWindow main, String path, String fileName) {
		main.getController().actionMenuFileLoad(path, fileName);
		main.getController().actionMenuHelpAutoML();
	}
}
