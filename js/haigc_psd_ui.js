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

        if (nodeData.name === "HAIGC_LoadPSD") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);

                const btnName = "选择PSD文件";
                const exists = this.widgets?.find(w => w.name === btnName);
                if (!exists) {
                    const btn = this.addWidget("button", btnName, "选择并上传", async () => {
                        const input = document.createElement("input");
                        input.type = "file";
                        input.accept = ".psd";
                        input.onchange = async () => {
                            const file = input.files?.[0];
                            if (!file) return;
                            
                            // Check file size (e.g. 100MB warning, though we try to support more)
                            // We will use chunked upload to bypass server limits
                            
                            try {
                                const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks
                                const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
                                const filename = file.name;
                                let psdPath = "";
                                
                                // Step 1: Initialize upload
                                const initForm = new FormData();
                                initForm.append("filename", filename);
                                initForm.append("total_chunks", totalChunks);
                                initForm.append("action", "init");
                                
                                const endpoints = ["/haigc_psd/upload_psd", "/haigc/upload_psd", "/haigc/upload"];
                                let endpoint = endpoints[0]; // Default to first
                                
                                // Helper to try endpoints
                                const tryFetch = async (formData) => {
                                    let lastErr;
                                    for (const ep of endpoints) {
                                        try {
                                            const r = await fetch(ep, { method: "POST", body: formData });
                                            if (r.ok) {
                                                endpoint = ep; // Stick to working endpoint
                                                return await r.json();
                                            }
                                            lastErr = await r.text();
                                        } catch (e) { lastErr = e; }
                                    }
                                    throw new Error(lastErr || "Upload failed");
                                };

                                const initData = await tryFetch(initForm);
                                const uploadId = initData.upload_id;

                                // Step 2: Upload chunks
                                for (let i = 0; i < totalChunks; i++) {
                                    const start = i * CHUNK_SIZE;
                                    const end = Math.min(file.size, start + CHUNK_SIZE);
                                    const chunk = file.slice(start, end);
                                    
                                    const chunkForm = new FormData();
                                    chunkForm.append("upload_id", uploadId);
                                    chunkForm.append("chunk_index", i);
                                    chunkForm.append("file", chunk, filename);
                                    chunkForm.append("action", "chunk");
                                    
                                    // Update button text to show progress
                                    btn.name = `上传中 ${(i/totalChunks*100).toFixed(0)}%`;
                                    this.setDirtyCanvas(true, true);
                                    
                                    await tryFetch(chunkForm);
                                }
                                
                                // Step 3: Finalize
                                const finishForm = new FormData();
                                finishForm.append("upload_id", uploadId);
                                finishForm.append("action", "finish");
                                const finalData = await tryFetch(finishForm);
                                psdPath = finalData.path;
                                
                                btn.name = btnName; // Restore button name

                                const modeWidget = this.widgets?.find(w => w.name === "输入模式");
                                if (modeWidget) {
                                    modeWidget.value = "文件";
                                    modeWidget.callback?.(modeWidget.value);
                                }

                                const pathWidget = this.widgets?.find(w => w.name === "PSD文件路径");
                                if (pathWidget) {
                                    pathWidget.value = psdPath;
                                    pathWidget.callback?.(pathWidget.value);
                                }

                                this.setDirtyCanvas(true, true);
                            } catch (e) {
                                console.error(e);
                                btn.name = "上传失败";
                                setTimeout(() => { btn.name = btnName; this.setDirtyCanvas(true,true); }, 2000);
                                alert(`PSD上传失败: ${e?.message || e}`);
                            }
                        };
                        input.click();
                    });
                    btn.serialize = false;

                    // const pathIndex = this.widgets?.findIndex(w => w.name === "PSD文件路径") ?? -1;
                    // if (pathIndex >= 0) {
                    //     const w = this.widgets.pop();
                    //     this.widgets.splice(pathIndex + 1, 0, w);
                    // }
                }

                return r;
            };
        }
	},
});
