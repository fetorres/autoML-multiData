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
 *
 */

package org.deidentifier.arx.gui.view.impl.menu;

import org.deidentifier.arx.gui.Controller;
import org.deidentifier.arx.gui.resources.Resources;
import org.deidentifier.arx.gui.view.SWTUtil;
import org.deidentifier.arx.gui.view.def.IDialog;
import org.eclipse.jface.dialogs.IMessageProvider;
import org.eclipse.jface.dialogs.TitleAreaDialog;
import org.eclipse.jface.window.Window;
import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.CTabFolder;
import org.eclipse.swt.custom.CTabItem;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Text;

/**
 * Added by Yunhui A class similar to DialogAbout, but displays the instructions
 * related to AutoML
 */
public class DialogAutoMLInfo extends TitleAreaDialog implements IDialog {

	/** TODO */
	private Image image;

	/**
	 * Constructor.
	 *
	 * @param parentShell
	 * @param controller
	 */
	public DialogAutoMLInfo(final Shell parentShell, final Controller controller) {
		super(parentShell);
		this.image = controller.getResources().getManagedImage("logo_small.png"); //$NON-NLS-1$
	}

	@Override
	public boolean close() {
		return super.close();
	}

	@Override
	protected void configureShell(Shell newShell) {
		super.configureShell(newShell);
		newShell.setImages(Resources.getIconSet(newShell.getDisplay()));
	}

	@Override
	// The OK button
	protected void createButtonsForButtonBar(final Composite parent) {

		// Create OK Button
		parent.setLayoutData(SWTUtil.createFillGridData());
		final Button okButton = createButton(parent, Window.OK, Resources.getMessage("AboutDialog.15"), true); //$NON-NLS-1$
		okButton.addSelectionListener(new SelectionAdapter() {
			@Override
			public void widgetSelected(final SelectionEvent e) {
				setReturnCode(Window.OK);
				close();
			}
		});
	}

	@Override
	protected Control createContents(Composite parent) {
		Control contents = super.createContents(parent);
		setTitle(Resources.getMessage("AutoML.9")); //$NON-NLS-1$
		setMessage(Resources.getMessage("AutoML.10"), IMessageProvider.INFORMATION); //$NON-NLS-1$
		if (image != null)
			setTitleImage(image); // $NON-NLS-1$
		return contents;
	}

	@Override
	protected Control createDialogArea(final Composite parent) {
		parent.setLayout(new GridLayout());

		// Folder
		CTabFolder folder = new CTabFolder(parent, SWT.BORDER);
		folder.setSimple(false);
		folder.setLayoutData(SWTUtil.createFillGridData());
		folder.setSelection(0);

		// Instructions
		CTabItem item1 = new CTabItem(folder, SWT.NULL);
		item1.setText("Instructions"); //$NON-NLS-1$
		final Text instr = new Text(folder, SWT.NONE | SWT.MULTI | SWT.V_SCROLL | SWT.BORDER);
		instr.setText(Resources.getMessage("AutoML.12"));
		instr.setEditable(false);
		instr.setLayoutData(SWTUtil.createFillGridData());
		item1.setControl(instr);

		// Privacy Models
		String messages = Resources.getMessage("AutoML.13a") + "\n" + Resources.getMessage("AutoML.13b") + "\n"
				+ Resources.getMessage("AutoML.13c") + "\n" + Resources.getMessage("AutoML.13d") + "\n"
				+ Resources.getMessage("AutoML.13e1") + "\n" + Resources.getMessage("AutoML.13e2") + "\n"
				+ Resources.getMessage("AutoML.13e3") + "\n" + Resources.getMessage("AutoML.13f") + "\n"
				+ Resources.getMessage("AutoML.13g") + "\n" + Resources.getMessage("AutoML.13h");
		CTabItem item2 = new CTabItem(folder, SWT.NULL);
		item2.setText("Privacy"); //$NON-NLS-1$
		final Text privacy = new Text(folder, SWT.NONE | SWT.MULTI | SWT.V_SCROLL | SWT.BORDER);
		privacy.setText(messages);
		privacy.setEditable(false);
		privacy.setLayoutData(SWTUtil.createFillGridData());
		item2.setControl(privacy);

		return parent;
	}

	@Override
	protected boolean isResizable() {
		return false;
	}
}
