# FRBpulse_injection
Inject fake generated Fast Radio Burst pulses into known background data to retrain FETCH.

Using these python files, fake generated Fast Radio Bursts can be injected in any real telescope background data. 

## Guide to Retrain Fetch
- Get the lilo background data from any telescope without bursts
- with `pulse_injection.py` we can inject a single pulse in this data and converting it to h5 candidate, which can then be checked for faults and mistakes
- with `param_test_heim_fetch.py` we can inject a gaussian multiple component pulse in the background data to convert it to possible h5 candidate.
- with `param_mass_parameter.py` we can run the `param_test_heim_fetch.py` code multiple times with random generated parameters. 
- the folder `inject_pulse` will retain all information of the previous run, but will get overwritten when a new burst is being made. 

With this, you are able to create thousand real .h5 candidates to retrain the convolutional neural network FETCH.  
