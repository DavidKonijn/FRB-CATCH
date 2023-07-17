# Fast Radio Burst Pulse Injection 

Inject fake generated Fast Radio Burst pulses into known background data to retrain FETCH.

The goal of this research is to find extragalactic, milliseconds-long pulses, dubbed Fast Radio Bursts, in the local Universe. By localising these FRBs in specific starburst galaxies, we get a comprehensive look into the origins of these mysterious bursts. As part of this project, I have retrained the Convolutional Neural Network FETCH, which improved the detection accuracy of the pipeline. Additionally, I collaborated with three major radio telescope departments, Toruń, Nançay, and Westerbork, to schedule observations and process the radio data with the improved pipeline.

Using these python files, fake generated Fast Radio Bursts can be injected in any real telescope background data. 

## Guide to Retrain Fetch
- Get the lilo background data from any telescope without bursts
- with `pulse_injection.py` we can inject a single pulse in this data and converting it to h5 candidate, which can then be checked for faults and mistakes
- with `param_test_heim_fetch.py` we can inject a gaussian multiple component pulse in the background data to convert it to possible h5 candidate.
- with `param_mass_parameter.py` we can run the `param_test_heim_fetch.py` code multiple times with random generated parameters. 
- the folder `inject_pulse` will retain all information of the previous run, but will get overwritten when a new burst is being made. 

With this, you are able to create thousand real .h5 candidates to retrain the convolutional neural network FETCH.  

With the box_##.py files, it is possible to find the basic burst parameters. The box_dm_time.py file acts as a safety net for FETCH.
