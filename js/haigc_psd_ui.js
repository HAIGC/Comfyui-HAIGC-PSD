import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "HAIGC.PSD.Download",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "HAIGC_SavePSD") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function(message) {
				onExecuted?.apply(this, arguments);
                
                // message contains the output from the python node
                // message.psd_filename should be available if we pass it
                
                if (message && message.psd_filename) {
                    const filename = message.psd_filename[0];
                    const subfolder = message.subfolder ? message.subfolder[0] : "";
                    const type = "output";
                    
                    // Add or update a button widget to download
                    // Since standard widgets are for inputs, we might need a custom method or just a DOM overlay
                    // But ComfyUI nodes are canvas based. 
                    // However, 'onExecuted' usually handles showing images.
                    // We can append a "Download PSD" button to the image preview area or add a button widget.
                    
                    // Simplest way: Add a button widget if it doesn't exist
                    const w_name = "Download PSD";
                    const existing = this.widgets?.find(w => w.name === w_name);
                    
                    if (!existing) {
                        const btn = this.addWidget("button", w_name, "Click to Download", () => {
                            const params = new URLSearchParams({
                                filename: filename,
                                subfolder: subfolder,
                                type: type
                            });
                            window.open(`/view?${params.toString()}`, "_blank");
                        });
                        btn.serialize = false; // Don't save this widget state
                    } else {
                        // Update callback closure with new filename if needed
                        existing.callback = () => {
                             const params = new URLSearchParams({
                                filename: filename,
                                subfolder: subfolder,
                                type: type
                            });
                            window.open(`/view?${params.toString()}`, "_blank");
                        }
                    }
                    
                    // Force redraw
                    this.setDirtyCanvas(true, true);
                }
			};
		}
	},
});
