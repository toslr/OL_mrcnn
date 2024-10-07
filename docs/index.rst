.. OL_mrcnn documentation master file, created by
   sphinx-quickstart on Mon Apr 15 14:13:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
====================================

This is an introduction guide to setup and use a Fiji plugin for oligodendrocyte segmentation.

Installation
====================================

Prerequisites: Python 3.9, Miniconda 3, Fiji

Downloads required: 

- model: https://github.com/toslr/OL_mrcnn

- plugin and weight files here_

.. _here: https://drive.google.com/drive/folders/1PIstT451WQIOS59vtHqkq8PTD-xO0Gj_?usp=sharing


1. From a terminal, place yourself in the ``OL_mrcnn-main`` model folder. Create a conda environment using the .yml file:

.. code-block::

   conda env create -f environment.yml

2. Place the model weights in the folder ``/logs``. These contain the original weights (trained on COCO dataset) and the weights trained on a custom dataset. Feel free to add your own weight files.

3. Open the ``config.txt`` file in the ``OL_segmentation`` folder and set:

    - the first line to the path to the environmental python. You can use the following command to find it and add ``/bin/python`` at the end of the path if not already there:
   
      .. code-block::

         conda info --envs
   
   
    - the second line to the path to the ``OL_mrcnn-main`` folder

4. Move the ``OL_segmentation`` plugin folder inside the Fiji plugins. Restart Fiji to take into account the changes.


Plugin User Guide
=================

Preprocessing 
~~~~~~~~~~~~~~

Before starting your analyses, your images need to be readable. Launch the "Preprocess images" options of the Fiji plugin. A window should pop.

- Select the folder you want to preprocess.
- Select the type of your images.
- Select the channels you want to take into account. This choice will affect the accuracy of the results. Below is an example of which channels give the best results.

.. figure:: /_static/wrong_channel_select.png
   :height: 300px
   :width: 360px
   :alt: alternative text

   Selecting only green and blue here will lead to bad results (regardless of contrast).

.. figure:: /_static/right_channel_select.png
   :height: 300px
   :width: 360px
   :alt: alternative text

   Better to select red and blue / red and green and blue.

- Select whether you want RGB or grayscale images. RGB is recommended for the analysis. 
- Hit "Ok". Your new folder of images will appear next to the former one under the name "oldername_norm"


Analysis
~~~~~~~~~
Select the "Run MRCNN" option of the Fiji plugin. A window should pop.

- Select the task you want to achieve. **The segmentation part is still under development. Please refer to the Python userguide for segmenting.**
- Choose the dataset you want to process.
- Select the weight file you want to use. Click Other to choose your own.
- Choose a name for the task. it will be the name of the results folder. 
- Adjust the confidence and non-maximum suppression thresholds.
- Choose whether you want to visualize and edit the results, 
- Choose if you want to save the crops.
- Hit "Ok".

If the results editor has been selected, the images will be displayed along with the ROIs. You can navigate between the images, edit, add or delete the ROIs. Click 'Finish' to save the changes.

The final results of your job will be saved under the folder ``results/jobName``.


Python User Guide
=================

Image cropping
--------------

- Add your dataset in the folder ``/data``
- OPTIONAL: preprocess your data with the ``preprocessing.ipynb`` notebook
- Configure the ``image_cropper.ipynb`` notebook:

   - ``DEVICE``: device to use for inference. Default value is 'cpu:0'.
   - ``detection_min_confidence``: minimum confidence level for the detections. Default value is 0.7.
   - ``detection_nms_threshold``: non-maximum suppression threshold. Eliminates the least confident detection when the IoU of 2 masks is above this value. Default value is 0.3.
   - ``weights_subpath``: subpath in the `/logs` folder to the weights file.
   - ``results_name``: name of the folder where the results will be saved.
   - ``test_dir``: name of the folder where the images are stored.
   - ``num_gpu``: number of GPUs to use for inference. Default value is 1.
   - ``num_img_per_gpu``: number of images to process in parallel on each GPU. Default value is 1.
   - ``VISUALIZE``: if True, displays the images with the detections. Default value is False.

- Run the notebook. The results will be saved in the folder ``/results/results_name``. 

Full pipeline
-------------

- In this setup, we run a first model to crop and classify objects in the images. Then we run a second model on the cropped images to get a refined mask.

- In the ``model_pipeline.ipynb`` notebook, configure the following parameters:

   - ``DEVICE``: device to use for inference. Default value is 'cpu:0'.
   - ``gpu_count_macro``: number of GPUs to use for the first model. Default value is 1.
   - ``num_img_per_gpu_macro``: number of images to process in parallel on each GPU for the first model. Default value is 1.
   - ``min_confidence_macro``: minimum confidence level for the detections in the first model. Default value is 0.7.
   - ``nms_threshold_macro``: non-maximum suppression threshold for the first model. Default value is 0.3.
   - ``nms_multiclass_macro``: non-maximum suppression threshold between classes for the first model. Default value is 0.3.
   - ``gpu_count_micro``: number of GPUs to use for the second model. Default value is 1.
   - ``num_img_per_gpu_micro``: number of images to process in parallel on each GPU for the second model. Default value is 1.
   - ``min_confidence_micro``: minimum confidence level for the detections in the second model. Default value is 0.7.
   - ``nms_threshold_micro``: non-maximum suppression threshold for the second model. Default value is 0.3.
   - ``MACRO_MODEL_SUBPATH``: subpath in the `/logs` folder to the weights file of the first model.
   - ``MICRO_MODEL_SUBPATH``: subpath in the `/logs` folder to the weights file of the second model.
   - ``RESULTS_NAME``: name of the folder where the results will be saved.
   - ``TEST_DIR``: name of the folder where the images are stored.
   - ``VISUALIZE``: if True, displays the images with the detections. Default value is False.

- Run the notebook. The results will be saved in the folder ``/results/RESULTS_NAME``.


Retraining your own model
-------------------------

Data structure
~~~~~~~~~~~~~~

- Create a ``/data`` folder in the root directory.
- Inside the ``/data`` directory, put your images in a folder named ``/imgs`` and your binary masks in a folder named ``/masks``. The name, size and format of the masks must match the images.
- In the ``roi_labels_to_json.py``script, configure the ``dir_path``in the `main()` function. Run in a terminal:

.. code-block::

   python roi_labels_to_json.py

- Move the label files to a ``jsons`` folder in the ``/data``directory.
- In the ``format_data.py`` script, configure the ``dir_path`` in the `main()` function. Configure the size the of the training / validation / test datasets (usually 0.6, 0.2, 0.2) Run in a terminal:

.. code-block::

   python format_data.py


Retraining a single class model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- In the ``custom.py`` script, configure the following:
   - ``GRAYSCALE``: if True, the model will be trained on grayscale images. Default value is False.
   - ``DATA_PATH``: path to the dataset. Default value is '/data'.
   - ``NAME``: name of the model.
   - ``GPU_COUNT``: number of GPUs to use. Default value is 1.
   - ``IMAGES_PER_GPU``: number of images to process in parallel on each GPU. Default value is 1.
   - ``NUM_CLASSES``: number of classes. Default value is 2.
   - ``EPOCHS``: number of epochs. Default value is 50.
   - ``STEPS PER EPOCH``: number of steps per epoch. Default value is 50.
   - ``LEARNING_RATE``: learning rate. Default value is 0.001.
   - ``LAYERS``: layers to train. Default value is 'heads'.
   - ``DETECTION_MIN_CONFIDENCE``: minimum confidence level for the detections. Default value is 0.7.
   - ``DEVICE``: device to use for training. Default value is 'cpu:0'.
   - ``MAX_GT_INSTANCES``: maximum number of instances in the ground truth. Default value is 100.
   - ``DETECTION_MAX_INSTANCES``: maximum number of instances in the detections. Default value is 35.
   - in the ``CustomDataset``class, modify or add lines :

   .. code-block::
      self.add_class(<NAME>, <class_number> , <class_name>)

- Run the script in a terminal:

.. code-block::

   python custom.py


Retraining a multi-class model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Same instructions as before but on the ``custom_multi.py`` script.

- Run the script in a terminal:

.. code-block::

   python custom_multi.py


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
