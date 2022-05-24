This file is a spiking conversion of the MNIST dataset. The conversion process is described in:

Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.  “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuromorphic Engineering,  special topic on Benchmarks and Challenges for Neuromorphic Engineering



Each example is a separate binary file consisting of a list of events. Each event occupies 40 bits arranged as described below:

bit 39 - 32: Xaddress (in pixels)
bit 31 - 24: Yaddress (in pixels)
bit 23: Polarity (0 for OFF, 1 for ON)
bit 22 - 0: Timestamp (in microseconds)


The filenames match the original MNIST dataset so that spike recordings inlcuded here can bebacktraced to the original images.

A Matlab function for "Read_Ndataset.m" is provided for reading these binary files into Matlab.

Additional Matlab functions as well as a Python module are available at: http://www.garrickorchard.com/code

The bias parameters used by the ATIS during recording are:
APSvrefL:  3050mV
APSvrefH:  3150mV
APSbiasOut: 750mV
APSbiasHyst: 620mV
CtrlbiasLP: 620mV
APSbiasTail: 700mV
CtrlbiasLBBuff: 950mV
TDbiasCas: 2000mV
CtrlbiasDelTD: 400mV
TDbiasDiffOff: 620mV
CtrlbiasSeqDelAPS: 320mV
TDbiasDiffOn: 780mV
CtrlbiasDelAPS: 350mV
TDbiasInv: 880mV
biasSendReqPdY: 850mV
TDbiasFo: 2950mV
biasSendReqPdX: 1150mV
TDbiasDiff: 700mV
CtrlbiasGB: 1050mV
TDbiasBulk: 2680mV
TDbiasReqPuY: 810mV
TDbiasRefr: 2900mV
TDbiasReqPuX: 1240mV
TDbiasPR: 3150mV
APSbiasReqPuY: 1100mV
APSbiasReqPuX: 820mV