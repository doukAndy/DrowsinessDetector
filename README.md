# DrowsynessDetector


### dataset
*Raw dataset downloaded from [figshare](https://figshare.com/articles/dataset/The_original_EEG_data_for_driver_fatigue_detection/5202739).

*The file structure has been organized as follows:
```
data/
├── data_figshare
│    ├── 1
│    │   ├── Fatigue state.cnt
│    │   └── Normal state.cnt
│    ├── 2
│    │   ├── Fatigue state.cnt
│    │   └── Normal state.cnt
│    ├── ...
│    └── 12
│        ├── Fatigue state.cnt
│        └── Normal state.cnt
└── readme.md
```

*Run process.get_data_tool to get raw data and label in .mat form.

### model
*[EEGNet](https://arxiv.org/abs/1611.08024): code heavily refers to the [official code in Keras](https://github.com/vlawhern/arl-eegmodels).

*[Conformer](https://ieeexplore.ieee.org/document/9991178), code heavily refers to the [official repo](https://github.com/eeyhsong/EEG-Conformer), which also gives the basic idea for this task.

*[Hierarchical Transformer](https://ieeexplore.ieee.org/document/10078473), a relatively new research without official open source.

### experiment
*Configurations for the whole experiment are set in ./experiment/config.py.

*In ./experiment/cross_subject.py, 

    **EEG signals from all subjects are used for training and validating (3:1) except the 1st and the 2nd subjects' (used for testing).

    **3 models are trained and saved and tested.

    **results and some figures are in ./result.
