File Formats
============

Configuration
^^^^^^^^^^^^^
A configuration file consists of the assignment of the following
variables. Any unassigned variable will hold its corresponding default
value.

* **use_gpu**         (default = 0)     1 to use GPU, 0 for CPU.
* **mult**            (default = 0)     1 to use the multiplicative method, 0 otherwise.
* **mult_exp**        (default = .25)   Multiplicative method exponent.
* **hf_method**       (default = 0)     1 to use HF Method, 0 otherwise.
* **mtiff**           (default = 0)     1 if data is in multipage tiff files.
* **frames**          (default = "")    

  Array of values in the format start frame, end frame, skip value. No value means all frames are to be processed. 
  frames being one value leads to the processing of that frame alone. Two values indicates the start and end frames 
  with 0 skipped. Three values indicates the start frame, end frame, and skip value.

                                      
* **calib_file_path** (default = "")    Path to calibration file, relative to config filepath.
* **images_path**     (default = "")    Path to data, relative to config filepath.

Example configuration file contents::

  use_gpu=0
  hf_method=1
  calib_file_path= ./pinhole_calibration_data/calibration_results/results_000001438970105.txt
  images_path= ./pinhole_data/set1_text/

  mtiff=0

  mult=0
  mult_exp=0.5
  #frames was omitted to indicate all frames be processed

