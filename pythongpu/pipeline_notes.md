causation entropy fmri adjacency matrix structure:

put fmri file into data file, organized as a new file for datasets. they are big so idk how storage will work necessarily? 
find way to import patientnumber.nifti
    something about a key that i got from the HCP website which let me put the data file into the server

fork: do we... (1) or (2)?

    (1) use some accepted neuroscience parcellation method -- 83 regions for example. i worry this would delete important data, since tens of thousands of voxels have to be mapped to 83 distinct nodes. also how do we guarantee everyone's brain is partitioned similarly? if its from a purely geometric set-border type of thing will that disregard individual anatomy? is that information encoded in the fmri data?

        - we need to then show a map of the brain partitioned into these 83 regions


    (2) find the optimal number of regions to observe lower scale dynamics -- similar to elbow method. what would the issue be if people had drastically different number of nodes?

        - output print how many nodes is optimal + a graph of this


ok so at this point we have our NODES!!! yay
    then we need to implement the entropy regression in order to create directed edges between points


        questions:

            - is there a way to turn directed edges into undirected edges?? idk if directed edges would be too complicated atp in the research


            - how do i define basis functions? what does that mean here?

output adjacency matrix
side quest for later? animate the graph being formed over the time series for demonstration