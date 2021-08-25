Shravan Chaudhari - Graph Neural Networks for End-to-End Particle Identification with the CMS Experiment (Boosted Top Jets)

## Summary
The training notebooks demonstrate the performance comparison of Graph Neural Networks (x1 granularity) and Convolutional Neural Networks (x3 granularity). As can be seen from the ROC curves and ROC AUC score, both GNN and CNN have around similar performance for specific channels. The channels used here are pT, ECAL and HCAL along with the coordinates [eta,phi] for GNNs and all the 8 channels (pT, ECAL, HCAL, dz, d0, BPIX1, BPIX2, BPIX3) for CNNs. 

## Results
1) CNN on 640K Boosted Top Jets (x3 granularity) - Test ROC AUC score: 0.976 +/- 0.01
2) GNN on 32K samples of Boosted Top Jets (x1 granularity with just energy channels and pT track) - Test AUC score: 0.971 +/- 0.01

## Future Objectives
The future goals of the project include scaling the Graph Neural Networks to more channels (pixel layers and tracker layers) and train CNNs & GNNs on a much larger dataset for developing a robust model. 
