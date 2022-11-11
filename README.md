# FRBpulse_injection
Inject generated FRB pulses into known background data 

Using these python files, fake generated Fast Radio Bursts can be injected in any real telescope background data. 

param_mass_parameter.py injects x bursts with all random properties between values that can be changed to the individual likings and creates h5 candidates from them. RFI is zapped using rfifind, and the burst is only logged when detected by Heimdall, 
