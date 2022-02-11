# img_viewer.py

import PySimpleGUI as sg
import os.path
import glob
import time
import itertools

import sys
import threading
import qptiff
import filterVessel
import coco

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'

class MainWindow ():
    def __init__(self):
        self.state = {}

    def start (self):
        self.window = sg.Window("BIIF Deep Image Classifier", self.getLayout(), finalize=True, icon=r'C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\src\biif_logo_white.ico')
        self.window["log"].reroute_stdout_to_here()
        #self.window["log"].reroute_stderr_to_here()

        self.window["log"].expand (expand_x = True)
        self.window["_TREE_"].expand (expand_x = True)
        self.window["progbar"].expand (expand_x = True)
        self.window.write_event_value('-init-',True) 

        # Run the Event Loop
        while True:
            event, values = self.window.read()
            self.values = values
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            # Folder name was filled in, make a list of files in the folder
            if event in ["-FOLDER-", "-Recursive-", "-Filter-", "-RGBImages-", "-qptiffImages-"] and values["-FOLDER-"].strip() != "":
                self.folder = values["-FOLDER-"]
                self.filter = values["-Filter-"]
                self.recursively = values["-Recursive-"]
                self.load_folder ()
            elif event == "-toImageJ-":
                self.window["-toQuPath-"].update(disabled=not values["-toImageJ-"])
            elif event == '-RUN-':
                self.DLthread = threading.Thread(target=self.runDL, daemon=True).start()
            elif event == "_TREE_":
                print ("{0} file(s) selected for processing.".format(len(self.window.Element('_TREE_').SelectedRows)))

            #elif event == "-CANCEL-":
            #    self.DLthread
            self.updateLayout ()

        self.window.close()
        
    def runDL(self):
        self.window["-RUN-"].update(disabled=True)
        self.window['progbar'].update_bar(10)
        max_prog = 1000
        
        try:
            processedFiles = self.window.Element('_TREE_').SelectedRows #self.load_folder ()
            
            print ("Importing Tensorflow module.")
            from maskrcnn_vessels import vessel
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            if self.values["-RGBImages-"]:
                cocoFilename = coco.ImagesToCOCO(processedFiles, self.folder)
                self.values["-ImagesToQuPath-"] = False

                self.window['progbar'].update_bar(max_prog/2)
                vessel.detect_images(
                    cocoFilename, 
                    model_dir=os.path.dirname(os.path.abspath(__file__)) + "/maskrcnn_vessels/logs/",
                    render_qupath_ROIs = False,
                    render_ImageJ_ROIs = self.values["-ImagesToImageJ-"],
                    render_RGB_labels = self.values["-ImagesToRGB-"]
                )
                self.window['progbar'].update_bar(max_prog)
            else:
                extractMethod = ""
                if self.values["-qptiff_annotations_only-"]:
                    extractMethod = "annot"
                elif self.values["-qptiff_annotations_all-"]:
                    extractMethod = "all"
                elif self.values["-qptiff_annotations_random-"]:
                    extractMethod = "random"
                elif self.values["-qptiff_annotations_all_CCL21-"]:
                    extractMethod = "all_CCL21"
                    self.values["-CCL21Mask-"] = True
                for qptiffIndex, qptiffFile in enumerate(processedFiles):
                    max_prog = (qptiffIndex+1) * max_prog/len(processedFiles)
                    print ("Processing {0}".format(qptiffFile))
                    outputDir = "{0}/DL_Results/".format(os.path.dirname(qptiffFile))
                    qptiffObject = qptiff.QPtiff(qptiffFile, outputDir)
                    qptiffObject.extractAnnotationsFromQptiff(
                        extractMethod=extractMethod, 
                        CCL21Mask=self.values["-CCL21Mask-"], 
                        fatMask=self.values["-FatMask-"], 
                        metastasisMask=self.values["-MetastasisMask-"]
                    )
                    qptiffObject.saveCOCO()
                    cocoFilename = qptiffObject.cocoFilename

                    self.window['progbar'].update_bar(max_prog/2)
                    
                    vessel.detect_images(
                        cocoFilename, 
                        model_dir=os.path.dirname(os.path.abspath(__file__)) + "/maskrcnn_vessels/logs/",
                        render_qupath_ROIs = self.values["-ImagesToQuPath-"],
                        render_ImageJ_ROIs = self.values["-ImagesToImageJ-"],
                        render_RGB_labels = self.values["-ImagesToRGB-"]
                    )
                    if (self.values["-ImagesToQuPath-"]):
                        filterVessel.filterVessels(outputDir, 600)
                    self.window['progbar'].update_bar(max_prog)
        except:
            import traceback
            traceback.print_exc()
        print ("Processing complete")
        self.window['progbar'].update_bar(1000)
        self.window["-RUN-"].update(disabled=False)
        

    def load_folder (self):
        returnedList = []

        def add_files_in_folder(parent, dirname):
            files = os.listdir(dirname)
            for f in files:
                fullname = os.path.join(dirname, f)
                if os.path.isdir(fullname) and self.recursively:            # if it's a folder, add folder and recurse
                    self.treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
                    add_files_in_folder(fullname, fullname)
                else:
                    filename, extension = os.path.splitext(f)
                    if (extension in [".png",".tif",".tiff"] and self.values["-RGBImages-"]) or (extension in [".qptiff"] and self.values["-qptiffImages-"]):
                        if (self.filter in f):
                            self.treedata.Insert(parent, fullname, f, values=[], icon=file_icon)
                            returnedList.append(fullname)
        
        self.treedata = sg.TreeData()
        add_files_in_folder("", self.folder)
        self.window['_TREE_'].Update(values=self.treedata)
        self.window.refresh()
        #if self.values["-RGBImages-"]:
        #    self.window["_FILELIST_"].update(values=tiff_files)
        #else:
        #    self.window["_FILELIST_"].update(values=qptiff_files)
        print ("{0} file(s) selected for processing.".format(len(returnedList)))
        for selected_row in self.window.Element('_TREE_').SelectedRows:
            print (self.window.Element('_TREE_').TreeData.tree_dict[selected_row].values)

        def key_to_id(tree, key):
            """
            Convert PySimplGUI element key to tkinter widget id.
            : Parameter
            key - key of PySimpleGUI element.
            : Return
            id - int, id of tkinter widget
            """
            dictionary = {v:k for k, v in tree.IdToKey.items()}
            return dictionary[key] if key in dictionary else None

        def select(tree, keys=[]):
            """
            Move the selection of node to node key.
            : Parameters
            key - str, key of node.
            """
            ids = []
            for key in keys:
                id_ = key_to_id(tree, key)
                print (id_)
                if id_:
                    tree.Widget.see(id_)
                    ids.append(id_)
            tree.Widget.selection_set(ids)
        #self.window["_TREE_"].Widget.selection_set(None)
        select(self.window["_TREE_"], keys=returnedList)

    def updateLayout (self):
        for key in ["-qptiff_annotations_only-", "-qptiff_annotations_all-", "-qptiff_annotations_random-", "-qptiff_annotations_all_CCL21-","-ImagesToQuPath-",
        "-CCL21Mask-","-FatMask-","-MetastasisMask-"]:
            self.window[key].update(disabled=self.values["-RGBImages-"])
            if self.values["-qptiff_annotations_all_CCL21-"] and key=="-CCL21Mask-":
                self.window[key].update(disabled=True)

    def getLayout (self):
        # First the window layout in 2 columns
        self.treedata = sg.TreeData()
        file_list_column = [
            [
                sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(),
            ],
            [
                sg.pin(sg.Frame(title='Input options', layout=[
                    [
                        sg.Checkbox("Browse folders recursively", enable_events=True, key="-Recursive-"),
                    ],
                    [
                        sg.Text("Filter input:"), sg.InputText(enable_events=True, key="-Filter-", size=(20,40)),
                    ],
                    [
                        sg.Radio("Process RGB tiff images", group_id="tiff_or_qptiff", enable_events=True, key="-RGBImages-"),
                    ],
                    [
                        sg.Radio("Process qptiff images", default=True, group_id="tiff_or_qptiff", enable_events=True, key="-qptiffImages-"),
                    ],
                    [
                        sg.Text('   '), sg.Radio("Only scanner annotations (from xml)", enable_events=True, group_id="qptiff_annotations", key="-qptiff_annotations_only-"),
                    ],
                    [
                        sg.Text('   '), sg.Radio("Random tiles from qptiff", enable_events=True, group_id="qptiff_annotations", key="-qptiff_annotations_random-"),
                    ],
                    [
                        sg.Text('   '), sg.Radio("All tiles containing CCL21+", enable_events=True, group_id="qptiff_annotations", key="-qptiff_annotations_all_CCL21-", default=True),
                    ],
                    [
                        sg.Text('   '), sg.Radio("All tiles", enable_events=True, group_id="qptiff_annotations", key="-qptiff_annotations_all-"),
                    ]
                ]))
            ],
            [
                sg.pin(sg.Frame(title='Mask options', layout=[
                    [
                        sg.Checkbox("Compute and use CCL21 mask", enable_events=True, key="-CCL21Mask-", default=True),
                    ],
                    [
                        sg.Checkbox("Compute and use Fat mask", enable_events=True, key="-FatMask-", default=True),
                    ],
                    [
                        sg.Checkbox("Compute and use Metastasis mask", enable_events=True, key="-MetastasisMask-", default=True),
                    ]
                ]))
            ],
            [
                sg.pin(sg.Frame(title='Output options', layout=[
                    [
                        sg.Checkbox("Export ImageJ ROIs", key="-ImagesToImageJ-", default=True),
                    ],
                    [
                        sg.Checkbox("Export CSV and QuPath ROIs", tooltip="Export QuPath ROIs", key="-ImagesToQuPath-", default=True),
                    ],
                    [
                        sg.Checkbox("Export RGB images with labels", tooltip="Export RGB images with labels", key="-ImagesToRGB-", default=True),
                    ],
                ]))
            ],
            [sg.Submit("Run deep learning segmentation", key="-RUN-", tooltip='Click to start Deep Learning segmentation')],
        ]

        # For now will only show the name of the file that was chosen
        image_viewer_column = [
            [sg.Text('Files:')],
            [
                #sg.Listbox(
                #    values=[], enable_events=True, size=(40, 20), key="-FILE LIST-", select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE 
                #)
                sg.Tree(data=self.treedata, headings=[], auto_size_columns=True, num_rows=15, col0_width=88, key='_TREE_', show_expanded=True, enable_events=True)
            ],
            [sg.Text('Logs:')],
            [
                #sg.Listbox(
                #    values=[], enable_events=True, size=(40, 20), key="-FILE LIST-", select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE 
                #)
                sg.Multiline(size=(75,15), key='log', autoscroll=True, write_only=True, font='Courier 10'),
            ],
            [
                sg.ProgressBar(1000, orientation='h', size=(70, 20), key='progbar'),
                #sg.Submit("Cancel", key="-CANCEL-", disabled=True, tooltip='Click to cancel processing'),
            ]
        ]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(file_list_column, vertical_alignment="top"),
                sg.VSeperator(),
                sg.Column(image_viewer_column,vertical_alignment="top"),
            ]
        ]
        return layout


if __name__ == "__main__":
    # execute only if run as a script
    mw = MainWindow()
    mw.start()